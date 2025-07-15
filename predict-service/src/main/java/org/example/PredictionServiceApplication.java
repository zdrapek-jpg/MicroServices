package org.example;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.web.client.RestTemplate;
import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.http.client.HttpComponentsClientHttpRequestFactory;

/**
 * Main application class for the Prediction Service.
 * This class uses @SpringBootApplication, which is a convenience annotation that adds:
 * - @Configuration: Tags the class as a source of bean definitions for the application context.
 * - @EnableAutoConfiguration: Tells Spring Boot to start adding beans based on classpath settings,
 * other beans, and various property settings.
 * - @ComponentScan: Tells Spring to look for other components, configurations, and services in the
 * 'com.example.predictionservice' package.
 */
@SpringBootApplication
public class PredictionServiceApplication {

    /**
     * The main method that starts the Spring Boot application.
     * @param args Command line arguments.
     */
    public static void main(String[] args) {
        // Set the server port for the Prediction Service to 8082 to avoid conflicts.
        // This can also be done in application.properties file.
        System.setProperty("server.port", "8082");
        SpringApplication.run(PredictionServiceApplication.class, args);
    }

    /**
     * Defines a RestTemplate bean with timeouts.
     * The deprecated setBufferRequestBody(false) call has been removed,
     * as modern Spring versions and underlying HTTP clients should handle
     * Content-Length header correctly by default for non-streaming requests.
     *
     * @param builder RestTemplateBuilder provided by Spring Boot.
     * @return Configured RestTemplate instance.
     */
    @Bean
    public RestTemplate restTemplate(RestTemplateBuilder builder) {
        return builder
                .setConnectTimeout(java.time.Duration.ofSeconds(5)) // 5 seconds to establish connection
                .setReadTimeout(java.time.Duration.ofSeconds(10))  // 10 seconds to read response
                // The HttpComponentsClientHttpRequestFactory is still used implicitly by RestTemplateBuilder
                // when Apache HttpClient is on the classpath.
                // The setBufferRequestBody(false) call is deprecated and removed.
                // The default behavior should now correctly send Content-Length for non-streaming bodies.
                .build();
    }
}

