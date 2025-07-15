package org.example;


import java.util.List;

/**
 * Represents the exact JSON structure expected by the external prediction API.
 * This class will be serialized into:
 * {
 * "user_data": [
 * "Henryk", 619, "France", ...
 * ]
 * }
 */
public class PredictionApiRequest {
    private List<Object> user_data; // This list will hold mixed types

    public PredictionApiRequest() {
    }

    public PredictionApiRequest(List<Object> user_data) {
        this.user_data = user_data;
    }

    // Getter and Setter
    public List<Object> getUser_data() {
        return user_data;
    }

    public void setUser_data(List<Object> user_data) {
        this.user_data = user_data;
    }

}