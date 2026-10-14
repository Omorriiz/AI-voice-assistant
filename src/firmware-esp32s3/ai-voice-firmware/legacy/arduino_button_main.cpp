
#include <Arduino.h>
#include <WiFi.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>
#include <driver/i2s.h>
#include <mbedtls/base64.h>

#include "config.h"

// ============ Значения по умолчанию (можно переопределить в config.h) ============
#ifndef SAMPLE_RATE
#define SAMPLE_RATE 16000
#endif
#ifndef MAX_RECORD_SECONDS
#define MAX_RECORD_SECONDS 10
#endif
#ifndef MAX_SAMPLES
#define MAX_SAMPLES (SAMPLE_RATE * MAX_RECORD_SECONDS)
#endif
#ifndef AUDIO_CHUNK_BYTES
#define AUDIO_CHUNK_BYTES 3200  // 100 мс PCM16 @ 16 kHz
#endif
#ifndef BUTTON_RECORD
#define BUTTON_RECORD 4
#endif
// INMP441 отдаёт 24 бита в 32-битном слоте. Сдвиг 16 = «как есть», 14 = +12 dB.
#ifndef MIC_SHIFT
#define MIC_SHIFT 14
#endif
// Канал INMP441: L/R пин на GND → левый. Если запись — тишина, попробуйте I2S_CHANNEL_FMT_ONLY_RIGHT.
#ifndef MIC_CHANNEL_FMT
#define MIC_CHANNEL_FMT I2S_CHANNEL_FMT_ONLY_LEFT
#endif

#if !defined(I2S_MIC_BCK) || !defined(I2S_MIC_WS) || !defined(I2S_MIC_DIN)
#error "Define I2S_MIC_BCK, I2S_MIC_WS and I2S_MIC_DIN (INMP441 pins) in config.h"
#endif

// ============ Пины и порты ============
constexpr i2s_port_t MIC_PORT = I2S_NUM_0;
constexpr i2s_port_t SPK_PORT = I2S_NUM_1;

// MAX98357A — подтверждено рабочим тестом (PIN_SET_1)
constexpr int SPK_BCLK_PIN = 5;
constexpr int SPK_LRC_PIN = 6;
constexpr int SPK_DIN_PIN = 7;

constexpr int DMA_FRAMES = 256;

static_assert(BUTTON_RECORD != SPK_BCLK_PIN && BUTTON_RECORD != SPK_LRC_PIN && BUTTON_RECORD != SPK_DIN_PIN &&
              BUTTON_RECORD != I2S_MIC_BCK && BUTTON_RECORD != I2S_MIC_WS && BUTTON_RECORD != I2S_MIC_DIN,
              "BUTTON_RECORD conflicts with an I2S pin");
static_assert(I2S_MIC_BCK != SPK_BCLK_PIN && I2S_MIC_BCK != SPK_LRC_PIN && I2S_MIC_BCK != SPK_DIN_PIN &&
              I2S_MIC_WS != SPK_BCLK_PIN && I2S_MIC_WS != SPK_LRC_PIN && I2S_MIC_WS != SPK_DIN_PIN &&
              I2S_MIC_DIN != SPK_BCLK_PIN && I2S_MIC_DIN != SPK_LRC_PIN && I2S_MIC_DIN != SPK_DIN_PIN,
              "INMP441 pins conflict with MAX98357A pins 5/6/7");

// ============ Глобальные объекты ============
WebSocketsClient webSocket;

// ============ Состояния ============
enum DeviceState {
    STATE_IDLE,
    STATE_RECORDING,
    STATE_PROCESSING,
    STATE_PLAYING,
    STATE_ERROR
};

DeviceState currentState = STATE_IDLE;

// ============ Буфер аудио ============
int16_t* audioBuffer = nullptr;
size_t audioBufferSize = 0;
bool isRecording = false;
unsigned long recordingStartTime = 0;

bool micReady = false;
bool speakerReady = false;

// ============ Тайминги ============
unsigned long lastReconnectAttempt = 0;
unsigned long stateTimeout = 0;
const unsigned long PROCESSING_TIMEOUT = 45000;
const unsigned long PLAYING_TIMEOUT = 30000;

// ============ Счётчики ============
size_t incomingAudioChunks = 0;

// ============ Прототипы ============
bool setupI2SMicrophone();
bool setupI2SSpeaker();
void connectToWiFi();
void connectToWebSocket();
void startRecording();
void stopRecording();
void recordAudioChunk();
void resetPlayback();
void playPCM16Chunk(const uint8_t* data, size_t len);
void finishPlayback();
void sendJsonMessage(JsonDocument& doc);
void webSocketEvent(WStype_t type, uint8_t* payload, size_t length);
void handleButton();

// ============================================================
// ИНИЦИАЛИЗАЦИЯ I2S ДЛЯ МИКРОФОНА (INMP441)
// ============================================================
bool setupI2SMicrophone() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,  // INMP441: 24 бита в 32-битном слоте
        .channel_format = MIC_CHANNEL_FMT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = DMA_FRAMES,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };

    i2s_pin_config_t pin_config = {};
    pin_config.bck_io_num = I2S_MIC_BCK;
    pin_config.ws_io_num = I2S_MIC_WS;
    pin_config.data_out_num = I2S_PIN_NO_CHANGE;
    pin_config.data_in_num = I2S_MIC_DIN;
    pin_config.mck_io_num = I2S_PIN_NO_CHANGE;

    esp_err_t err = i2s_driver_install(MIC_PORT, &i2s_config, 0, NULL);
    if (err == ESP_OK) err = i2s_set_pin(MIC_PORT, &pin_config);
    if (err != ESP_OK) {
        Serial.printf("[I2S] Микрофон: ошибка %s\n", esp_err_to_name(err));
        return false;
    }

    Serial.printf("[I2S] Микрофон INMP441: BCK=%d WS=%d DIN=%d\n", I2S_MIC_BCK, I2S_MIC_WS, I2S_MIC_DIN);
    return true;
}

// ============================================================
// ИНИЦИАЛИЗАЦИЯ I2S ДЛЯ ДИНАМИКА (MAX98357A) — как в рабочем тесте
// ============================================================
bool setupI2SSpeaker() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = DMA_FRAMES,
        .use_apll = false,
        .tx_desc_auto_clear = true,  // тишина, когда нечего играть
        .fixed_mclk = 0
    };

    i2s_pin_config_t pin_config = {};
    pin_config.bck_io_num = SPK_BCLK_PIN;
    pin_config.ws_io_num = SPK_LRC_PIN;
    pin_config.data_out_num = SPK_DIN_PIN;
    pin_config.data_in_num = I2S_PIN_NO_CHANGE;
    pin_config.mck_io_num = I2S_PIN_NO_CHANGE;

    esp_err_t err = i2s_driver_install(SPK_PORT, &i2s_config, 0, NULL);
    if (err == ESP_OK) err = i2s_set_pin(SPK_PORT, &pin_config);
    if (err != ESP_OK) {
        Serial.printf("[I2S] Динамик: ошибка %s\n", esp_err_to_name(err));
        return false;
    }
    i2s_zero_dma_buffer(SPK_PORT);

    Serial.printf("[I2S] Усилитель MAX98357A: BCLK=%d LRC=%d DIN=%d\n", SPK_BCLK_PIN, SPK_LRC_PIN, SPK_DIN_PIN);
    return true;
}

// Короткий сигнал при старте — сразу слышно, что динамик работает
void playStartupBeep() {
    static int16_t buffer[DMA_FRAMES];
    const float step = 2.0f * PI * 880.0f / SAMPLE_RATE;
    float phase = 0.0f;
    const int total = SAMPLE_RATE / 8;  // 125 мс
    for (int offset = 0; offset < total; offset += DMA_FRAMES) {
        for (int i = 0; i < DMA_FRAMES; i++) {
            buffer[i] = (int16_t)(6000.0f * sinf(phase));
            phase += step;
        }
        size_t written = 0;
        i2s_write(SPK_PORT, buffer, sizeof(buffer), &written, pdMS_TO_TICKS(200));
    }
    finishPlayback();
}

// ============================================================
// ПОДКЛЮЧЕНИЕ К WI-FI
// ============================================================
void connectToWiFi() {
    Serial.printf("\n[WiFi] Подключаюсь к %s...", WIFI_SSID);

    WiFi.disconnect(true, true);
    delay(500);
    WiFi.mode(WIFI_STA);
    delay(100);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 40) {
        delay(500);
        Serial.print(".");
        attempts++;
    }

    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\n[WiFi] Подключено!");
        Serial.printf("[WiFi] IP: %s\n", WiFi.localIP().toString().c_str());
        connectToWebSocket();
    } else {
        Serial.println("\n[WiFi] Ошибка подключения");
        currentState = STATE_ERROR;
    }
}

// ============================================================
// ПОДКЛЮЧЕНИЕ К WEBSOCKET
// ============================================================
void connectToWebSocket() {
    Serial.printf("[WS] Подключаюсь к ws://%s:%d%s\n",
                  WS_SERVER_IP, WS_SERVER_PORT, WS_SERVER_PATH);

    webSocket.begin(WS_SERVER_IP, WS_SERVER_PORT, WS_SERVER_PATH);
    webSocket.onEvent(webSocketEvent);
    webSocket.setReconnectInterval(5000);
    webSocket.enableHeartbeat(15000, 3000, 2);
}

// ============================================================
// ОТПРАВКА JSON
// ============================================================
void sendJsonMessage(JsonDocument& doc) {
    String message;
    serializeJson(doc, message);
    webSocket.sendTXT(message);
}

// ============================================================
// СТАРТ ЗАПИСИ
// ============================================================
void startRecording() {
    if (currentState != STATE_IDLE) {
        Serial.println("[REC] Нельзя начать — не IDLE");
        return;
    }
    if (!micReady) {
        Serial.println("[REC] Микрофон не инициализирован");
        currentState = STATE_ERROR;
        return;
    }

    if (audioBuffer != nullptr) {
        free(audioBuffer);
        audioBuffer = nullptr;
    }

    audioBuffer = (int16_t*)ps_malloc(MAX_SAMPLES * sizeof(int16_t));
    if (audioBuffer == nullptr) {
        audioBuffer = (int16_t*)malloc(MAX_SAMPLES * sizeof(int16_t));
    }

    if (audioBuffer == nullptr) {
        Serial.println("[REC] Ошибка выделения буфера!");
        currentState = STATE_ERROR;
        return;
    }

    // Выбросить старые данные, накопившиеся в DMA, пока запись не шла
    i2s_zero_dma_buffer(MIC_PORT);
    static int32_t drain[DMA_FRAMES];
    size_t got = 0;
    for (int i = 0; i < 16; i++) {
        if (i2s_read(MIC_PORT, drain, sizeof(drain), &got, 0) != ESP_OK || got == 0) break;
    }

    audioBufferSize = 0;
    isRecording = true;
    recordingStartTime = millis();
    currentState = STATE_RECORDING;

    Serial.printf("[REC] Начало записи (свободно PSRAM: %d KB)\n",
                  ESP.getFreePsram() / 1024);

    JsonDocument doc;
    doc["type"] = "start_recording";
    sendJsonMessage(doc);
}

// ============================================================
// СТОП ЗАПИСИ + ОТПРАВКА
// ============================================================
void stopRecording() {
    if (currentState != STATE_RECORDING) return;

    isRecording = false;
    float duration = audioBufferSize / (float)SAMPLE_RATE;
    Serial.printf("[REC] Запись завершена: %.2f сек, %d сэмплов\n",
                  duration, audioBufferSize);

    if (!webSocket.isConnected()) {
        Serial.println("[WS] Не подключён!");
        free(audioBuffer);
        audioBuffer = nullptr;
        currentState = STATE_ERROR;
        return;
    }

    if (audioBufferSize == 0) {
        Serial.println("[REC] Пустая запись");
        free(audioBuffer);
        audioBuffer = nullptr;
        currentState = STATE_IDLE;
        return;
    }

    currentState = STATE_PROCESSING;
    stateTimeout = millis();

    const uint8_t* raw = (const uint8_t*)audioBuffer;
    size_t bytesToSend = audioBufferSize * sizeof(int16_t);
    size_t offset = 0;
    int chunkIndex = 0;

    static unsigned char b64[((AUDIO_CHUNK_BYTES + 2) / 3) * 4 + 1];

    Serial.printf("[WS] Отправка %d байт аудио...\n", bytesToSend);

    while (offset < bytesToSend) {
        size_t remaining = bytesToSend - offset;
        size_t chunkSize = (remaining < AUDIO_CHUNK_BYTES) ? remaining : (size_t)AUDIO_CHUNK_BYTES;

        size_t b64Len = 0;
        if (mbedtls_base64_encode(b64, sizeof(b64), &b64Len, raw + offset, chunkSize) != 0) {
            Serial.println("[WS] Ошибка base64");
            break;
        }

        JsonDocument doc;
        doc["type"] = "audio_chunk";
        doc["index"] = chunkIndex;
        doc["data"] = (const char*)b64;

        String jsonStr;
        serializeJson(doc, jsonStr);

        if (!webSocket.sendTXT(jsonStr)) {
            Serial.println("[WS] Ошибка отправки чанка");
            break;
        }

        offset += chunkSize;
        chunkIndex++;
        delay(10);
        webSocket.loop();
    }

    JsonDocument stopDoc;
    stopDoc["type"] = "stop_recording";
    sendJsonMessage(stopDoc);

    Serial.printf("[WS] Отправлено %d чанков\n", chunkIndex);

    free(audioBuffer);
    audioBuffer = nullptr;
    audioBufferSize = 0;
}

// ============================================================
// ЗАПИСЬ ЧАНКА (32-битные сэмплы INMP441 → PCM16)
// ============================================================
void recordAudioChunk() {
    if (!isRecording || audioBuffer == nullptr) return;

    static int32_t raw[DMA_FRAMES];
    size_t bytesRead = 0;

    esp_err_t err = i2s_read(MIC_PORT, raw, sizeof(raw), &bytesRead, pdMS_TO_TICKS(20));
    if (err != ESP_OK || bytesRead == 0) return;

    const size_t samples = bytesRead / sizeof(int32_t);

    if (audioBufferSize + samples < MAX_SAMPLES) {
        for (size_t i = 0; i < samples; i++) {
            int32_t s = raw[i] >> MIC_SHIFT;
            if (s > 32767) s = 32767;
            if (s < -32768) s = -32768;
            audioBuffer[audioBufferSize + i] = (int16_t)s;
        }
        audioBufferSize += samples;
    } else {
        Serial.println("[REC] Буфер заполнен");
        stopRecording();
        return;
    }

    if (millis() - recordingStartTime > MAX_RECORD_SECONDS * 1000UL) {
        Serial.println("[REC] Максимум времени записи");
        stopRecording();
    }
}

// ============================================================
// ВОСПРОИЗВЕДЕНИЕ (моно PCM16 → MAX98357A)
// ============================================================
static int16_t playSamples[DMA_FRAMES];
static size_t playSampleCount = 0;
static uint8_t playCarryByte = 0;  // нечётный байт, разрезанный между чанками
static bool playHasCarry = false;

void flushPlaySamples() {
    if (playSampleCount == 0) return;
    const size_t bytes = playSampleCount * sizeof(int16_t);
    size_t written = 0;
    esp_err_t err = i2s_write(SPK_PORT, playSamples, bytes, &written, pdMS_TO_TICKS(1000));
    if (err != ESP_OK || written != bytes) {
        Serial.printf("[PLAY] Ошибка i2s_write: %s (%u/%u)\n",
                      esp_err_to_name(err), (unsigned)written, (unsigned)bytes);
    }
    playSampleCount = 0;
}

inline void pushPlaySample(int16_t sample) {
    playSamples[playSampleCount] = sample;
    if (++playSampleCount == DMA_FRAMES) flushPlaySamples();
}

void resetPlayback() {
    playSampleCount = 0;
    playHasCarry = false;
}

void playPCM16Chunk(const uint8_t* data, size_t len) {
    if (!speakerReady || len == 0) return;

    size_t i = 0;
    if (playHasCarry) {
        pushPlaySample((int16_t)(playCarryByte | (data[0] << 8)));
        playHasCarry = false;
        i = 1;
    }
    for (; i + 1 < len; i += 2) {
        pushPlaySample((int16_t)(data[i] | (data[i + 1] << 8)));
    }
    if (i < len) {
        playCarryByte = data[i];
        playHasCarry = true;
    }
    flushPlaySamples();
}

// Дописать тишину, чтобы хвост ответа доиграл из DMA и не обрезался
void finishPlayback() {
    flushPlaySamples();
    playHasCarry = false;
    memset(playSamples, 0, sizeof(playSamples));
    for (int i = 0; i < 8; i++) {
        size_t written = 0;
        i2s_write(SPK_PORT, playSamples, sizeof(playSamples), &written, pdMS_TO_TICKS(200));
    }
}

// ============================================================
// WEBSOCKET EVENT
// ============================================================
void webSocketEvent(WStype_t type, uint8_t* payload, size_t length) {
    switch (type) {
        case WStype_DISCONNECTED:
            Serial.println("[WS] Отключён");
            if (currentState != STATE_RECORDING) {
                currentState = STATE_ERROR;
            }
            break;

        case WStype_CONNECTED:
            Serial.println("[WS] Подключён!");
            currentState = STATE_IDLE;
            incomingAudioChunks = 0;
            break;

        case WStype_TEXT: {
            JsonDocument doc;
            DeserializationError err = deserializeJson(doc, payload, length);
            if (err) {
                Serial.printf("[WS] JSON ошибка: %s\n", err.c_str());
                return;
            }

            String msgType = doc["type"] | "";

            if (msgType == "hello") {
                Serial.printf("[WS] Сервер: %s\n",
                              (const char*)(doc["message"] | ""));
            }
            else if (msgType == "text_response") {
                String text = doc["text"] | "";
                Serial.printf("[LLM] %s\n", text.c_str());
            }
            else if (msgType == "audio_response_start") {
                incomingAudioChunks = 0;
                resetPlayback();
                currentState = STATE_PLAYING;
                stateTimeout = millis();
                Serial.println("[PLAY] Начало воспроизведения...");
            }
            else if (msgType == "audio_response_chunk") {
                const char* b64 = doc["data"] | "";
                const size_t b64Len = strlen(b64);
                if (b64Len == 0) return;

                const size_t maxLen = (b64Len / 4) * 3 + 3;
                uint8_t* audioData = (uint8_t*)malloc(maxLen);
                if (!audioData) return;

                size_t decodedLen = 0;
                if (mbedtls_base64_decode(audioData, maxLen, &decodedLen,
                                          (const unsigned char*)b64, b64Len) == 0 &&
                    decodedLen > 0) {
                    playPCM16Chunk(audioData, decodedLen);
                    incomingAudioChunks++;
                    stateTimeout = millis();
                } else {
                    Serial.println("[PLAY] Ошибка base64");
                }

                free(audioData);
            }
            else if (msgType == "audio_response_end") {
                finishPlayback();
                Serial.printf("[PLAY] Завершено (%d чанков)\n",
                              incomingAudioChunks);
                currentState = STATE_IDLE;
            }
            break;
        }
        default:
            break;
    }
}

// ============================================================
// КНОПКА
// ============================================================
void handleButton() {
    static unsigned long lastDebounce = 0;
    static int lastState = HIGH;

    int state = digitalRead(BUTTON_RECORD);

    if (state == LOW && lastState == HIGH && millis() - lastDebounce > 300) {
        lastDebounce = millis();

        if (currentState == STATE_IDLE) {
            startRecording();
        } else if (currentState == STATE_RECORDING) {
            stopRecording();
        } else if (currentState == STATE_ERROR) {
            currentState = STATE_IDLE;
        }
    }

    lastState = state;
}

// ============================================================
// SETUP
// ============================================================
void setup() {
    Serial.begin(115200);
    delay(1000);

    Serial.println("\n\n========================================");
    Serial.println("Нильс — ESP32-S3 + INMP441 + MAX98357A");
    Serial.println("========================================");
    Serial.printf("Flash: %d MB\n", ESP.getFlashChipSize() / (1024 * 1024));
    Serial.printf("PSRAM: %d KB\n", ESP.getPsramSize() / 1024);
    Serial.printf("Free heap: %d KB\n", ESP.getFreeHeap() / 1024);
    Serial.println("========================================\n");

    pinMode(BUTTON_RECORD, INPUT_PULLUP);

    micReady = setupI2SMicrophone();
    speakerReady = setupI2SSpeaker();
    if (speakerReady) playStartupBeep();

    connectToWiFi();

    if (currentState != STATE_ERROR) {
        currentState = STATE_IDLE;
        Serial.println("\n[MAIN] Готов. Нажмите кнопку для записи.\n");
    }
}

// ============================================================
// LOOP
// ============================================================
void loop() {
    webSocket.loop();
    handleButton();

    if (currentState == STATE_RECORDING) {
        recordAudioChunk();
    }

    if (currentState == STATE_PROCESSING &&
        millis() - stateTimeout > PROCESSING_TIMEOUT) {
        Serial.println("[MAIN] Таймаут обработки");
        currentState = STATE_IDLE;
    }

    if (currentState == STATE_PLAYING &&
        millis() - stateTimeout > PLAYING_TIMEOUT) {
        Serial.println("[MAIN] Таймаут воспроизведения");
        resetPlayback();
        i2s_zero_dma_buffer(SPK_PORT);
        currentState = STATE_IDLE;
    }

    if (WiFi.status() != WL_CONNECTED &&
        currentState != STATE_ERROR &&
        millis() - lastReconnectAttempt > 30000) {
        lastReconnectAttempt = millis();
        Serial.println("[WiFi] Переподключение...");
        connectToWiFi();
    }

    // Во время записи i2s_read() сам ждёт данные — лишняя задержка привела бы к потере сэмплов
    if (currentState != STATE_RECORDING) delay(10);
}
