/**
 * Нильс — прошивка ESP32-S3 (ESP-IDF + esp-sr)
 *
 * Функции:
 * - Постоянное прослушивание микрофона INMP441 и поиск ключевого слова «Привет ESP»
 *   (esp-sr WakeNet10, модель wn10_ru_privetesp) — прямо на ESP32, без сервера
 * - После слова: сигнал, запись команды до паузы в речи (VAD esp-sr)
 * - Отправка PCM16 16kHz mono по WebSocket на сервер (тот же протокол, что и раньше)
 * - Воспроизведение ответа через MAX98357A
 * - Состояния: LISTENING → RECORDING → SENDING → PROCESSING → PLAYING → LISTENING
 *
 * Протокол WebSocket не изменился:
 *   → start_recording, audio_chunk {index, data(base64)}, stop_recording
 *   ← hello, text_response, audio_response_start, audio_response_chunk {data}, audio_response_end
 *
 * Динамик настроен как в рабочем тесте MAX98357A: 16 бит, 16 кГц, пины I2S_AMP_* из config.h.
 * Все настройки (Wi-Fi, сервер, пины) — в include/config.h.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "nvs_flash.h"
#include "driver/i2s_std.h"
#include "esp_websocket_client.h"
#include "cJSON.h"
#include "mbedtls/base64.h"

#include "esp_afe_sr_models.h"
#include "esp_afe_config.h"
#include "esp_wn_iface.h"
#include "esp_vad.h"
#include "model_path.h"

#include "config.h"

// ============ Значения по умолчанию (можно переопределить в config.h) ============
#ifndef SAMPLE_RATE
#define SAMPLE_RATE 16000
#endif
#if SAMPLE_RATE != 16000
#error "esp-sr works only at 16 kHz: set SAMPLE_RATE 16000 in config.h"
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
#ifndef MIC_SHIFT
#define MIC_SHIFT 14  // INMP441: 24 бита в 32-битном слоте. 16 = как есть, 14 = +12 dB
#endif
#ifndef MIC_USE_RIGHT_CHANNEL
#define MIC_USE_RIGHT_CHANNEL 0  // 0: L/R пин INMP441 на GND, 1: на 3.3 V
#endif

#if !defined(I2S_MIC_BCK) || !defined(I2S_MIC_WS) || !defined(I2S_MIC_DIN)
#error "Define I2S_MIC_BCK, I2S_MIC_WS and I2S_MIC_DIN (INMP441 pins) in config.h"
#endif
#if I2S_MIC_BCK < 0 || I2S_MIC_WS < 0 || I2S_MIC_DIN < 0
#error "Set the real INMP441 pins in config.h"
#endif

// ============ Пины динамика (рабочий тест MAX98357A: 5/6/7) ============
#ifndef I2S_AMP_BCLK
#define I2S_AMP_BCLK 5
#endif
#ifndef I2S_AMP_LRC
#define I2S_AMP_LRC 6
#endif
#ifndef I2S_AMP_DIN
#define I2S_AMP_DIN 7
#endif
#define SPK_BCLK_PIN I2S_AMP_BCLK
#define SPK_LRC_PIN  I2S_AMP_LRC
#define SPK_DIN_PIN  I2S_AMP_DIN

#if I2S_MIC_BCK == SPK_BCLK_PIN || I2S_MIC_BCK == SPK_LRC_PIN || I2S_MIC_BCK == SPK_DIN_PIN || \
    I2S_MIC_WS == SPK_BCLK_PIN || I2S_MIC_WS == SPK_LRC_PIN || I2S_MIC_WS == SPK_DIN_PIN ||    \
    I2S_MIC_DIN == SPK_BCLK_PIN || I2S_MIC_DIN == SPK_LRC_PIN || I2S_MIC_DIN == SPK_DIN_PIN
#error "INMP441 pins conflict with MAX98357A pins (I2S_AMP_*)"
#endif

// ============ Распознавание речи ============
#define WAKE_WORD_MODEL      "privetesp"  // wn10_ru_privetesp
#define VAD_END_SILENCE_MS   700    // пауза, после которой команда считается законченной
#define WAKE_GRACE_MS        (VAD_END_SILENCE_MS + 300)  // хвост самого «Привет ESP» не считается командой
#define NO_SPEECH_TIMEOUT_MS 6000   // после слова ничего не сказали — вернуться к ожиданию

// ============ Тайминги ============
#define PROCESSING_TIMEOUT_MS 45000
#define PLAYING_TIMEOUT_MS    30000

static const char* TAG = "alfred";

// ============ Состояния ============
typedef enum {
    ST_LISTENING,   // ждём «Привет ESP»
    ST_RECORDING,   // пишем команду
    ST_SENDING,     // отправляем запись на сервер
    ST_PROCESSING,  // ждём ответ сервера
    ST_PLAYING      // играем ответ
} device_state_t;

static volatile device_state_t s_state = ST_LISTENING;
static volatile uint32_t s_state_since_ms = 0;
static volatile uint32_t s_last_audio_ms = 0;
static volatile bool s_resume_request = false;  // вернуться к прослушиванию (выполняет detect_task)

// ============ Объекты ============
static i2s_chan_handle_t s_mic = NULL;
static i2s_chan_handle_t s_spk = NULL;
static esp_websocket_client_handle_t s_ws = NULL;
static const esp_afe_sr_iface_t* s_afe = NULL;
static esp_afe_sr_data_t* s_afe_data = NULL;
static TaskHandle_t s_sender_task = NULL;
static bool s_ws_started = false;

// ============ Буфер записи (PSRAM) ============
static int16_t* s_rec_buf = NULL;
static volatile size_t s_rec_samples = 0;
static bool s_speech_started = false;

// ============ Приём WebSocket ============
static char* s_rx_buf = NULL;
static size_t s_rx_cap = 0;
static size_t s_incoming_chunks = 0;

static inline uint32_t now_ms(void) { return (uint32_t)(esp_timer_get_time() / 1000); }

static void set_state(device_state_t state) {
    s_state = state;
    s_state_since_ms = now_ms();
}

// ============================================================
// ДИНАМИК (MAX98357A)
// ============================================================
static void speaker_init(void) {
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_1, I2S_ROLE_MASTER);
    chan_cfg.auto_clear = true;  // тишина, когда нечего играть
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, &s_spk, NULL));

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = SPK_BCLK_PIN,
            .ws = SPK_LRC_PIN,
            .dout = SPK_DIN_PIN,
            .din = I2S_GPIO_UNUSED,
            .invert_flags = {.mclk_inv = false, .bclk_inv = false, .ws_inv = false},
        },
    };
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_BOTH;  // моно-сэмпл в оба канала: MAX98357A слышит его в любом режиме
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(s_spk, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(s_spk));
    ESP_LOGI(TAG, "Динамик MAX98357A: BCLK=%d LRC=%d DIN=%d", SPK_BCLK_PIN, SPK_LRC_PIN, SPK_DIN_PIN);
}

static void speaker_write(const int16_t* samples, size_t count) {
    size_t written = 0;
    const size_t bytes = count * sizeof(int16_t);
    esp_err_t err = i2s_channel_write(s_spk, samples, bytes, &written, 1000);
    if (err != ESP_OK || written != bytes) {
        ESP_LOGW(TAG, "i2s write: %s (%u/%u)", esp_err_to_name(err), (unsigned)written, (unsigned)bytes);
    }
}

static void play_beep(int freq_hz, int ms) {
    static int16_t buf[256];
    const float step = 2.0f * 3.14159265f * freq_hz / SAMPLE_RATE;
    float phase = 0.0f;
    const int total = SAMPLE_RATE * ms / 1000;
    for (int offset = 0; offset < total; offset += 256) {
        const int n = (total - offset) < 256 ? (total - offset) : 256;
        for (int i = 0; i < n; i++) {
            buf[i] = (int16_t)(6000.0f * sinf(phase));
            phase += step;
        }
        speaker_write(buf, n);
    }
}

// ============================================================
// ВОСПРОИЗВЕДЕНИЕ ОТВЕТА (моно PCM16)
// ============================================================
static int16_t s_play_buf[256];
static size_t s_play_count = 0;
static uint8_t s_play_carry = 0;  // нечётный байт, разрезанный между чанками
static bool s_play_has_carry = false;

static void play_flush(void) {
    if (s_play_count == 0) return;
    speaker_write(s_play_buf, s_play_count);
    s_play_count = 0;
}

static inline void play_push(int16_t sample) {
    s_play_buf[s_play_count++] = sample;
    if (s_play_count == 256) play_flush();
}

static void play_reset(void) {
    s_play_count = 0;
    s_play_has_carry = false;
}

static void play_pcm16(const uint8_t* data, size_t len) {
    size_t i = 0;
    if (len == 0) return;
    if (s_play_has_carry) {
        play_push((int16_t)(s_play_carry | (data[0] << 8)));
        s_play_has_carry = false;
        i = 1;
    }
    for (; i + 1 < len; i += 2) play_push((int16_t)(data[i] | (data[i + 1] << 8)));
    if (i < len) {
        s_play_carry = data[i];
        s_play_has_carry = true;
    }
    play_flush();
}

// Дописать тишину, чтобы хвост ответа доиграл из DMA и не обрезался
static void play_finish(void) {
    play_flush();
    s_play_has_carry = false;
    memset(s_play_buf, 0, sizeof(s_play_buf));
    for (int i = 0; i < 8; i++) speaker_write(s_play_buf, 256);
}

// ============================================================
// МИКРОФОН (INMP441)
// ============================================================
static void mic_init(void) {
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &s_mic));

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = I2S_MIC_BCK,
            .ws = I2S_MIC_WS,
            .dout = I2S_GPIO_UNUSED,
            .din = I2S_MIC_DIN,
            .invert_flags = {.mclk_inv = false, .bclk_inv = false, .ws_inv = false},
        },
    };
    std_cfg.slot_cfg.slot_mask = MIC_USE_RIGHT_CHANNEL ? I2S_STD_SLOT_RIGHT : I2S_STD_SLOT_LEFT;
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(s_mic, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(s_mic));
    ESP_LOGI(TAG, "Микрофон INMP441: BCK=%d WS=%d DIN=%d", I2S_MIC_BCK, I2S_MIC_WS, I2S_MIC_DIN);
}

// ============================================================
// ESP-SR: WakeNet «Привет ESP» + VAD
// ============================================================
static void afe_init(void) {
    srmodel_list_t* models = esp_srmodel_init("model");
    if (models == NULL) {
        ESP_LOGE(TAG, "Модели esp-sr не найдены в разделе \"model\" (прошейте через idf.py flash)");
        abort();
    }

    char* wn_name = esp_srmodel_filter(models, "wn", WAKE_WORD_MODEL);
    if (wn_name == NULL) {
        ESP_LOGE(TAG, "Модель wn10_ru_privetesp не найдена: включите CONFIG_SR_WN_WN10_RU_PRIVETESP");
        abort();
    }

    afe_config_t* cfg = afe_config_init("M", models, AFE_TYPE_SR, AFE_MODE_LOW_COST);
    cfg->wakenet_init = true;
    cfg->wakenet_model_name = wn_name;
    cfg->wakenet_mode = DET_MODE_90;
    cfg->aec_init = false;  // нет опорного канала динамика
    cfg->se_init = false;   // один микрофон
    cfg->vad_init = true;
    cfg->vad_min_noise_ms = VAD_END_SILENCE_MS;

    s_afe = esp_afe_handle_from_config(cfg);
    s_afe_data = s_afe->create_from_config(cfg);
    afe_config_free(cfg);
    if (s_afe_data == NULL) {
        ESP_LOGE(TAG, "Не удалось создать AFE");
        abort();
    }
    ESP_LOGI(TAG, "Ключевое слово: %s", wn_name);
}

// Микрофон → AFE
static void feed_task(void* arg) {
    const int chunk = s_afe->get_feed_chunksize(s_afe_data);
    const int nch = s_afe->get_feed_channel_num(s_afe_data);
    int32_t* raw = heap_caps_malloc(chunk * sizeof(int32_t), MALLOC_CAP_INTERNAL);
    int16_t* feed = heap_caps_calloc(chunk * nch, sizeof(int16_t), MALLOC_CAP_INTERNAL);
    assert(raw && feed);

    for (;;) {
        size_t got = 0;
        if (i2s_channel_read(s_mic, raw, chunk * sizeof(int32_t), &got, portMAX_DELAY) != ESP_OK) continue;
        const int n = (int)(got / sizeof(int32_t));
        for (int i = 0; i < chunk; i++) {
            int32_t s = i < n ? (raw[i] >> MIC_SHIFT) : 0;
            if (s > 32767) s = 32767;
            if (s < -32768) s = -32768;
            feed[i * nch] = (int16_t)s;
        }
        s_afe->feed(s_afe_data, feed);
    }
}

// Только из detect_task: AFE не трогаем из нескольких задач одновременно
static void resume_listening(void) {
    s_afe->reset_buffer(s_afe_data);
    s_afe->enable_wakenet(s_afe_data);
    set_state(ST_LISTENING);
    ESP_LOGI(TAG, "Слушаю. Скажите «Привет ESP»");
}

static void start_command_recording(void) {
    if (s_ws == NULL || !esp_websocket_client_is_connected(s_ws)) {
        ESP_LOGW(TAG, "Ключевое слово услышано, но сервер не подключён");
        return;
    }
    s_afe->disable_wakenet(s_afe_data);
    s_rec_samples = 0;
    s_speech_started = false;
    set_state(ST_RECORDING);
    ESP_LOGI(TAG, "«Привет ESP»! Говорите команду...");
    play_beep(880, 120);
}

static void record_frame(const afe_fetch_result_t* r) {
    const uint32_t elapsed = now_ms() - s_state_since_ms;

    size_t n = r->data_size / sizeof(int16_t);
    if (s_rec_samples + n > MAX_SAMPLES) n = MAX_SAMPLES - s_rec_samples;
    memcpy(s_rec_buf + s_rec_samples, r->data, n * sizeof(int16_t));
    s_rec_samples += n;

    if (r->vad_state == VAD_SPEECH && elapsed > WAKE_GRACE_MS) s_speech_started = true;

    if (s_speech_started && r->vad_state == VAD_SILENCE) {
        ESP_LOGI(TAG, "Конец фразы: %.2f сек", s_rec_samples / (float)SAMPLE_RATE);
    } else if (s_rec_samples >= MAX_SAMPLES) {
        ESP_LOGI(TAG, "Максимум времени записи");
    } else if (!s_speech_started && elapsed > NO_SPEECH_TIMEOUT_MS) {
        ESP_LOGI(TAG, "Команды не было — отмена");
        play_beep(440, 120);
        resume_listening();
        return;
    } else {
        return;
    }

    set_state(ST_SENDING);
    xTaskNotifyGive(s_sender_task);
}

// AFE → ключевое слово / запись команды / таймауты
static void detect_task(void* arg) {
    ESP_LOGI(TAG, "Слушаю. Скажите «Привет ESP»");
    for (;;) {
        afe_fetch_result_t* r = s_afe->fetch(s_afe_data);
        if (r == NULL || r->ret_value == ESP_FAIL) continue;

        if (s_resume_request) {
            s_resume_request = false;
            resume_listening();
            continue;
        }

        switch (s_state) {
            case ST_LISTENING:
                if (r->wakeup_state == WAKENET_DETECTED) start_command_recording();
                break;
            case ST_RECORDING:
                record_frame(r);
                break;
            case ST_PROCESSING:
                if (now_ms() - s_state_since_ms > PROCESSING_TIMEOUT_MS) {
                    ESP_LOGW(TAG, "Таймаут обработки");
                    resume_listening();
                }
                break;
            case ST_PLAYING:
                if (now_ms() - s_last_audio_ms > PLAYING_TIMEOUT_MS) {
                    ESP_LOGW(TAG, "Таймаут воспроизведения");
                    play_reset();
                    resume_listening();
                }
                break;
            default:
                break;  // ST_SENDING: аудио игнорируется, пока идёт отправка
        }
    }
}

// ============================================================
// ОТПРАВКА ЗАПИСИ
// ============================================================
static bool ws_send(const char* text, int len) {
    if (!esp_websocket_client_is_connected(s_ws)) return false;
    return esp_websocket_client_send_text(s_ws, text, len, pdMS_TO_TICKS(5000)) >= 0;
}

static void sender_task(void* arg) {
    static unsigned char b64[((AUDIO_CHUNK_BYTES + 2) / 3) * 4 + 1];
    static char json[sizeof(b64) + 64];

    for (;;) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

        const uint8_t* raw = (const uint8_t*)s_rec_buf;
        const size_t total = s_rec_samples * sizeof(int16_t);
        ESP_LOGI(TAG, "Отправка %u байт аудио...", (unsigned)total);

        const char* start = "{\"type\":\"start_recording\"}";
        bool ok = ws_send(start, strlen(start));
        int index = 0;
        for (size_t offset = 0; ok && offset < total; offset += AUDIO_CHUNK_BYTES) {
            const size_t size = (total - offset) < AUDIO_CHUNK_BYTES ? (total - offset) : AUDIO_CHUNK_BYTES;
            size_t b64_len = 0;
            if (mbedtls_base64_encode(b64, sizeof(b64), &b64_len, raw + offset, size) != 0) {
                ok = false;
                break;
            }
            const int len = snprintf(json, sizeof(json), "{\"type\":\"audio_chunk\",\"index\":%d,\"data\":\"%s\"}",
                                     index, (const char*)b64);
            ok = ws_send(json, len);
            index++;
        }

        if (!ok) {
            ESP_LOGW(TAG, "Ошибка отправки — возвращаюсь к прослушиванию");
            s_resume_request = true;
            continue;
        }

        set_state(ST_PROCESSING);  // до stop_recording: ответ может прийти сразу
        const char* stop = "{\"type\":\"stop_recording\"}";
        if (!ws_send(stop, strlen(stop))) {
            s_resume_request = true;
            continue;
        }
        ESP_LOGI(TAG, "Отправлено %d чанков, жду ответ", index);
    }
}

// ============================================================
// ПРИЁМ СООБЩЕНИЙ СЕРВЕРА
// ============================================================
static void handle_server_message(const char* text, size_t len) {
    cJSON* root = cJSON_ParseWithLength(text, len);
    if (root == NULL) {
        ESP_LOGW(TAG, "JSON ошибка");
        return;
    }
    const cJSON* type = cJSON_GetObjectItemCaseSensitive(root, "type");
    const char* t = cJSON_IsString(type) ? type->valuestring : "";

    if (strcmp(t, "hello") == 0) {
        const cJSON* msg = cJSON_GetObjectItemCaseSensitive(root, "message");
        ESP_LOGI(TAG, "Сервер: %s", cJSON_IsString(msg) ? msg->valuestring : "");
    } else if (strcmp(t, "text_response") == 0) {
        const cJSON* msg = cJSON_GetObjectItemCaseSensitive(root, "text");
        ESP_LOGI(TAG, "[LLM] %s", cJSON_IsString(msg) ? msg->valuestring : "");
    } else if (strcmp(t, "audio_response_start") == 0) {
        s_incoming_chunks = 0;
        play_reset();
        s_last_audio_ms = now_ms();
        set_state(ST_PLAYING);
        ESP_LOGI(TAG, "Начало воспроизведения...");
    } else if (strcmp(t, "audio_response_chunk") == 0 && s_state == ST_PLAYING) {
        const cJSON* data = cJSON_GetObjectItemCaseSensitive(root, "data");
        if (cJSON_IsString(data) && data->valuestring[0] != '\0') {
            const size_t b64_len = strlen(data->valuestring);
            const size_t max_len = (b64_len / 4) * 3 + 3;
            uint8_t* pcm = malloc(max_len);
            size_t pcm_len = 0;
            if (pcm && mbedtls_base64_decode(pcm, max_len, &pcm_len,
                                             (const unsigned char*)data->valuestring, b64_len) == 0) {
                play_pcm16(pcm, pcm_len);
                s_incoming_chunks++;
                s_last_audio_ms = now_ms();
            } else {
                ESP_LOGW(TAG, "Ошибка base64");
            }
            free(pcm);
        }
    } else if (strcmp(t, "audio_response_end") == 0) {
        play_finish();
        ESP_LOGI(TAG, "Воспроизведение завершено (%u чанков)", (unsigned)s_incoming_chunks);
        s_resume_request = true;
    }
    cJSON_Delete(root);
}

static void ws_event(void* arg, esp_event_base_t base, int32_t id, void* event_data) {
    esp_websocket_event_data_t* d = (esp_websocket_event_data_t*)event_data;
    switch (id) {
        case WEBSOCKET_EVENT_CONNECTED:
            ESP_LOGI(TAG, "WebSocket подключён");
            break;
        case WEBSOCKET_EVENT_DISCONNECTED:
        case WEBSOCKET_EVENT_ERROR:
            ESP_LOGW(TAG, "WebSocket отключён");
            if (s_state == ST_PROCESSING || s_state == ST_PLAYING) s_resume_request = true;
            break;
        case WEBSOCKET_EVENT_DATA: {
            if (d->op_code != 0x01 && d->op_code != 0x00) break;  // только текст (и его продолжение)
            if (d->payload_len <= 0 || d->data_len <= 0) break;
            // Большие сообщения приходят частями: собираем по payload_offset
            if (d->payload_offset == 0 && s_rx_cap < (size_t)d->payload_len + 1) {
                free(s_rx_buf);
                s_rx_cap = d->payload_len + 1;
                s_rx_buf = heap_caps_malloc(s_rx_cap, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
                if (s_rx_buf == NULL) {
                    s_rx_cap = 0;
                    ESP_LOGE(TAG, "Нет памяти для сообщения %d байт", d->payload_len);
                    break;
                }
            }
            if (s_rx_buf == NULL || (size_t)(d->payload_offset + d->data_len) >= s_rx_cap) break;
            memcpy(s_rx_buf + d->payload_offset, d->data_ptr, d->data_len);
            if (d->payload_offset + d->data_len >= d->payload_len) {
                s_rx_buf[d->payload_len] = '\0';
                handle_server_message(s_rx_buf, d->payload_len);
            }
            break;
        }
        default:
            break;
    }
}

static void websocket_start(void) {
    static char uri[128];
    snprintf(uri, sizeof(uri), "ws://%s:%d%s", WS_SERVER_IP, WS_SERVER_PORT, WS_SERVER_PATH);
    ESP_LOGI(TAG, "WebSocket: %s", uri);

    esp_websocket_client_config_t cfg = {
        .uri = uri,
        .buffer_size = 16384,
        .task_stack = 8192,
        .reconnect_timeout_ms = 5000,
        .network_timeout_ms = 10000,
        .ping_interval_sec = 15,
    };
    s_ws = esp_websocket_client_init(&cfg);
    esp_websocket_register_events(s_ws, WEBSOCKET_EVENT_ANY, ws_event, NULL);
    esp_websocket_client_start(s_ws);
}

// ============================================================
// WI-FI
// ============================================================
static void wifi_event(void* arg, esp_event_base_t base, int32_t id, void* data) {
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGW(TAG, "Wi-Fi отключён, переподключаюсь...");
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* e = (ip_event_got_ip_t*)data;
        ESP_LOGI(TAG, "Wi-Fi подключён, IP: " IPSTR, IP2STR(&e->ip_info.ip));
        if (!s_ws_started) {
            s_ws_started = true;
            websocket_start();  // дальше клиент переподключается сам
        }
    }
}

static void wifi_init(void) {
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t init_cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&init_cfg));
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, wifi_event, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, wifi_event, NULL));

    wifi_config_t cfg = {0};
    strncpy((char*)cfg.sta.ssid, WIFI_SSID, sizeof(cfg.sta.ssid));
    strncpy((char*)cfg.sta.password, WIFI_PASSWORD, sizeof(cfg.sta.password));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &cfg));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_LOGI(TAG, "Wi-Fi: подключаюсь к %s", WIFI_SSID);
}

// ============================================================
// MAIN
// ============================================================
void app_main(void) {
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "Нильс — ESP32-S3 + esp-sr «Привет ESP»");
    ESP_LOGI(TAG, "PSRAM свободно: %u KB", (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));
    ESP_LOGI(TAG, "========================================");

    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);

    s_rec_buf = heap_caps_malloc(MAX_SAMPLES * sizeof(int16_t), MALLOC_CAP_SPIRAM);
    if (s_rec_buf == NULL) {
        ESP_LOGE(TAG, "Нет PSRAM для буфера записи");
        abort();
    }

    speaker_init();
    mic_init();
    afe_init();

    xTaskCreatePinnedToCore(sender_task, "sender", 6144, NULL, 4, &s_sender_task, 1);
    xTaskCreatePinnedToCore(feed_task, "afe_feed", 4096, NULL, 5, NULL, 0);
    xTaskCreatePinnedToCore(detect_task, "afe_detect", 8192, NULL, 5, NULL, 1);

    play_beep(660, 100);  // устройство запущено
    wifi_init();
}
