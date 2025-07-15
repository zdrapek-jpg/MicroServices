package org.example;



/**
 * DTO to mirror the OrderWithUser object from OrderService, used for deserialization
 * within PredictionService. This includes the nested UserDetailsDTO.
 */
public class OrderDetailsDTO {


    private Integer id;
    private Integer userId;
    private String product;
    private UserDetailsDTO user; // Nested UserDetailsDTO

    // Constructors
    public OrderDetailsDTO() {
    }

    public OrderDetailsDTO(Integer id, Integer userId, String product, UserDetailsDTO user) {
        this.id = id;
        this.userId = userId;
        this.product = product;
        this.user = user;
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

    public UserDetailsDTO getUser() {
        return user;
    }

    public void setUser(UserDetailsDTO user) {
        this.user = user;
    }
}