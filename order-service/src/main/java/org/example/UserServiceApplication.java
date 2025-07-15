package org.example;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * Main application class for the User Service.
 * This class uses @SpringBootApplication, which is a convenience annotation that adds:
 * - @Configuration: Tags the class as a source of bean definitions for the application context.
 * - @EnableAutoConfiguration: Tells Spring Boot to start adding beans based on classpath settings,
 * other beans, and various property settings. For example, if spring-webmvc is on the classpath,
 * it flags the application as a web application and sets up a DispatcherServlet.
 * - @ComponentScan: Tells Spring to look for other components, configurations, and services in the
 * 'com.example.userservice' package, allowing it to find controllers, services, etc.
 */
@SpringBootApplication
public class UserServiceApplication {

    /**
     * The main method that starts the Spring Boot application.
     * @param args Command line arguments.
     */
    public static void main(String[] args) {
        SpringApplication.run(UserServiceApplication.class, args);
    }

}
