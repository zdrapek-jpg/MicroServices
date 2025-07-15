package org.example;


import org.springframework.web.bind.annotation.*;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * REST Controller for managing user-related operations.
 * This class provides endpoints to retrieve user information.
 */
@RestController // Marks this class as a REST controller, handling incoming web requests.
@RequestMapping("/users") // Maps all requests starting with /users to this controller.
public class UserController {

    // In-memory data store for demonstration purposes.
    // In a real application, this would be a database or another data source.
    private final Map<Integer, User> users = new ConcurrentHashMap<>();
    private final AtomicInteger idCounter = new AtomicInteger();

    /**
     * Constructor to initialize some dummy user data.
     */
    public UserController() {
        users.put(1, new User(1, "Alice"));
        users.put(2, new User(2, "Bob"));
        users.put(3, new User(3, "Charlie"));
    }

    /**
     * Retrieves a list of all users.
     *
     * @return A list of User objects.
     */
    @GetMapping // Maps GET requests to /users
    public List<User> getAllUsers() {
        System.out.println("Fetching all users...");
        return Arrays.asList(users.values().toArray(new User[0]));
    }

    /**
     * Retrieves a user by their ID.
     *
     * @param id The ID of the user to retrieve.
     * @return The User object corresponding to the given ID, or null if not found.
     */
    @GetMapping("/{id}") // Maps GET requests to /users/{id}
    public User getUserById(@PathVariable Integer id) {
        System.out.println("Fetching user with ID: " + id);
        return users.get(id);
    }
    @PostMapping("/add")
    public User addUser(@RequestBody User user) {
        int newId = idCounter.incrementAndGet();
        user.setId(newId);
        users.put(newId, user);
        System.out.println("Added new user: " + user.getName() + " with ID: " + newId);
        return user;
    }

    /**
     * Simple inner class representing a User.
     * In a real application, this would be a separate POJO (Plain Old Java Object) or Entity.
     */

    public static class User {
        private Integer id;
        private String name;

        public User(Integer id, String name) {
            this.id = id;
            this.name = name;
        }

        // Getters and Setters (required for JSON serialization/deserialization)
        public Integer getId() {
            return id;
        }

        public void setId(Integer id) {
            this.id = id;
        }

        public String getName() {
            return name;
        }

        public void setName(String name) {
            this.name = name;
        }
    }
}
