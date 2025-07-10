package org.example;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.web.client.RestTemplate;

/**
 * Main application class for the Order Service.
 * This class uses @SpringBootApplication, which is a convenience annotation that adds:
 * - @Configuration: Tags the class as a source of bean definitions for the application context.
 * - @EnableAutoConfiguration: Tells Spring Boot to start adding beans based on classpath settings,
 * other beans, and various property settings. For example, if spring-webmvc is on the classpath,
 * it flags the application as a web application and sets up a DispatcherServlet.
 * - @ComponentScan: Tells Spring to look for other components, configurations, and services in the
 * 'com.example.orderservice' package, allowing it to find controllers, services, etc.
 */
@SpringBootApplication
public class OrderServiceApplication {

    /**
     * The main method that starts the Spring Boot application.
     * @param args Command line arguments.
     */
    public static void main(String[] args) {
        // Set the server port for the Order Service to 8081 to avoid conflict with User Service (8080).
        // This can also be done in application.properties file.
        System.setProperty("server.port", "8081");
        SpringApplication.run(OrderServiceApplication.class, args);
    }

    /**
     * Defines a RestTemplate bean.
     * RestTemplate is a synchronous HTTP client for making REST calls.
     * It's used here to communicate with the UserService.
     * In newer Spring versions, WebClient is often preferred for reactive applications.
     * @return A new instance of RestTemplate.
     */
    @Bean
    public RestTemplate restTemplate() {
        return new RestTemplate();
    }
}
