package org.example;


/**
 * Represents the combined request body for the /orchestrate-prediction endpoint.
 * It contains data for the external prediction API and an order ID for the OrderService.
 */
public class CombinedPredictionRequest {
    private UserPredictionData userPredictionData;
    private Integer orderId;

    // Constructors
    public CombinedPredictionRequest() {
    }

    public CombinedPredictionRequest(UserPredictionData userPredictionData, Integer orderId) {
        this.userPredictionData = userPredictionData;
        this.orderId = orderId;
    }

    // Getters and Setters
    public UserPredictionData getUserPredictionData() {
        return userPredictionData;
    }

    public void setUserPredictionData(UserPredictionData userPredictionData) {
        this.userPredictionData = userPredictionData;
    }

    public Integer getOrderId() {
        return orderId;
    }

    public void setOrderId(Integer orderId) {
        this.orderId = orderId;
    }
}
