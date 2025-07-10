package org.example;


/**
 * Represents the combined response returned by the /orchestrate-prediction endpoint.
 * It includes the prediction result and the order details (with user info).
 */
public class CombinedPredictionResponse {
    private Object predictionResult; // Use Object to handle various JSON structures from external API
    private OrderDetailsDTO orderDetails;

    // Constructors
    public CombinedPredictionResponse() {
    }

    public CombinedPredictionResponse(Object predictionResult, OrderDetailsDTO orderDetails) {
        this.predictionResult = predictionResult;
        this.orderDetails = orderDetails;
    }

    // Getters and Setters
    public Object getPredictionResult() {
        return predictionResult;
    }

    public void setPredictionResult(Object predictionResult) {
        this.predictionResult = predictionResult;
    }

    public OrderDetailsDTO getOrderDetails() {
        return orderDetails;
    }

    public void setOrderDetails(OrderDetailsDTO orderDetails) {
        this.orderDetails = orderDetails;
    }
}
