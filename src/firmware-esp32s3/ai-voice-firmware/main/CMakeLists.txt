idf_component_register(
    SRCS "main.c"
    INCLUDE_DIRS "." "../include"
    REQUIRES esp_wifi esp_netif esp_event nvs_flash esp_timer esp_driver_i2s json mbedtls
)
