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


    @GetMapping // Maps GET requests to /orders
    public List<Order> getAllOrders() {
        System.out.println("Fetching all orders...");
        return Arrays.asList(orders.values().toArray(new Order[0]));
    }


    @GetMapping("/{id}")
    public Order getOrderById(@PathVariable Integer id) {
        System.out.println(" order with ID: " + id);
        return orders.get(id);
    }


    @GetMapping("/{id}/withUser")
    public OrderWithUser getOrderWithUser(@PathVariable Integer id) {
        System.out.println("Fetching order with ID: " + id + " and associated user.");
        Order order = orders.get(id);
        if (order == null) {
            return null;
        }


        UserController.User user = restTemplate.getForObject(
                USER_SERVICE_URL + order.getUserId(),
                UserController.User.class
        );


        return new OrderWithUser(order.getId(), order.getUserId(), order.getProduct(), user);
    }

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
        public UserController.User getUser() {
            return user;
        }

        public void setUser(UserController.User user) {
            this.user = user;
        }
    }
}