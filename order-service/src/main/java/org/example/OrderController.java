package org.example;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestTemplate;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

<<<<<<< HEAD

@RestController
@RequestMapping("/orders")
public class OrderController {


    private final Map<Integer, Order> orders = new ConcurrentHashMap<>();

    @Autowired
    private RestTemplate restTemplate;

    private static final String USER_SERVICE_URL = "http://localhost:8080/users/";


    public OrderController() {
        orders.put(101, new Order(101, 1, "Loan"));
        orders.put(102, new Order(102, 2, "Mortgage"));
        orders.put(103, new Order(103, 1, "Loan investment"));
    }


=======
/**
 * REST Controller for managing order-related operations.
 * This class provides endpoints to retrieve order information and
 * demonstrates inter-service communication by calling the UserService.
 */
@RestController // Marks this class as a REST controller, handling incoming web requests.
@RequestMapping("/orders") // Maps all requests starting with /orders to this controller.
public class OrderController {

    // In-memory data store for demonstration purposes.
    // In a real application, this would be a database or another data source.
    private final Map<Integer, Order> orders = new ConcurrentHashMap<>();

    // Autowired RestTemplate for making HTTP calls to other services.
    @Autowired
    private RestTemplate restTemplate;

    // URL of the User Service. In a real application, this would be configured
    // via a discovery service (e.g., Eureka, Consul) or externalized properties.
    private static final String USER_SERVICE_URL = "http://localhost:8080/users/";

    /**
     * Constructor to initialize some dummy order data.
     */
    public OrderController() {
        orders.put(101, new Order(101, 1, "Loan"));
        orders.put(102, new Order(102, 2, "mortgage"));
        orders.put(103, new Order(103, 1, "loan investment"));
    }

    /**
     * Retrieves a list of all orders.
     *
     * @return A list of Order objects.
     */
>>>>>>> 9db48ff01881fc7dced27e50501f42d2fe8f55f8
    @GetMapping // Maps GET requests to /orders
    public List<Order> getAllOrders() {
        System.out.println("Fetching all orders...");
        return Arrays.asList(orders.values().toArray(new Order[0]));
    }

<<<<<<< HEAD

    @GetMapping("/{id}")
    public Order getOrderById(@PathVariable Integer id) {
        System.out.println(" order with ID: " + id);
        return orders.get(id);
    }


    @GetMapping("/{id}/withUser")
=======
    /**
     * Retrieves an order by its ID.
     *
     * @param id The ID of the order to retrieve.
     * @return The Order object corresponding to the given ID, or null if not found.
     */
    @GetMapping("/{id}") // Maps GET requests to /orders/{id}
    public Order getOrderById(@PathVariable Integer id) {
        System.out.println("Fetching order with ID: " + id);
        return orders.get(id);
    }

    /**
     * Retrieves an order by its ID and enriches it with user details
     * by calling the UserService.
     *
     * @param id The ID of the order to retrieve.
     * @return An Order object with an embedded User object.
     */
    @GetMapping("/{id}/withUser") // Maps GET requests to /orders/{id}/withUser
>>>>>>> 9db48ff01881fc7dced27e50501f42d2fe8f55f8
    public OrderWithUser getOrderWithUser(@PathVariable Integer id) {
        System.out.println("Fetching order with ID: " + id + " and associated user.");
        Order order = orders.get(id);
        if (order == null) {
<<<<<<< HEAD
            return null;
        }


=======
            return null; // Or throw a custom exception
        }

        // Call the UserService to get user details
        // The second argument to getForObject is the expected response type.
        // It will automatically deserialize the JSON response into a User object.
>>>>>>> 9db48ff01881fc7dced27e50501f42d2fe8f55f8
        UserController.User user = restTemplate.getForObject(
                USER_SERVICE_URL + order.getUserId(),
                UserController.User.class
        );

<<<<<<< HEAD

        return new OrderWithUser(order.getId(), order.getUserId(), order.getProduct(), user);
    }

=======
        // Create a new object that combines order and user details
        return new OrderWithUser(order.getId(), order.getUserId(), order.getProduct(), user);
    }

    /**
     * Simple inner class representing an Order.
     */
>>>>>>> 9db48ff01881fc7dced27e50501f42d2fe8f55f8
    public static class Order {
        private Integer id;
        private Integer userId;
        private String product;

        public Order(Integer id, Integer userId, String product) {
            this.id = id;
            this.userId = userId;
            this.product = product;
        }

        // Getters and Setters
        public Integer getId() {
            return id;
        }

        public void setId(Integer id) {
            this.id = id;
        }

        public Integer getUserId() {
            return userId;
        }

        public void setUserId(Integer userId) {
            this.userId = userId;
        }

        public String getProduct() {
            return product;
        }

        public void setProduct(String product) {
            this.product = product;
        }
    }

    /**
     * A combined object to return order details along with user details.
     */
    public static class OrderWithUser extends Order {
        private UserController.User user;

        public OrderWithUser(Integer id, Integer userId, String product, UserController.User user) {
            super(id, userId, product);
            this.user = user;
        }
<<<<<<< HEAD
=======

        // Getter for user
>>>>>>> 9db48ff01881fc7dced27e50501f42d2fe8f55f8
        public UserController.User getUser() {
            return user;
        }

        public void setUser(UserController.User user) {
            this.user = user;
        }
    }
}