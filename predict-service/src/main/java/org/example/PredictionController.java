package org.example;


import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpEntity; // Import HttpEntity
import org.springframework.http.HttpHeaders; // Import HttpHeaders
import org.springframework.http.MediaType; // Import MediaType
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.web.client.ResourceAccessException;
import com.fasterxml.jackson.databind.ObjectMapper; // Import ObjectMapper for JSON serialization

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * REST Controller for orchestrating calls to an external prediction API and the OrderService.
 * This service accepts a combined request, fetches data from multiple sources,
 * and returns a consolidated response.
 */
@RestController // Marks this class as a REST controller.
@RequestMapping("/orchestrate-prediction") // Maps requests to this controller.
public class PredictionController {

    // Autowired RestTemplate for making HTTP calls.
    @Autowired
    private RestTemplate restTemplate;

    // ObjectMapper for manual JSON serialization
    private final ObjectMapper objectMapper = new ObjectMapper();

    // URL of the external prediction API.
    private static final String EXTERNAL_PREDICT_API_URL = "http://127.0.0.1:8000/predict/";
    // URL of the Order Service endpoint that provides order with user details.
    private static final String ORDER_SERVICE_WITH_USER_URL = "http://localhost:8081/orders/{id}/withUser";

    /**
     * Handles POST requests to /orchestrate-prediction.
     * Takes a CombinedPredictionRequest as input, calls the OrderService and
     * the external prediction API, and then combines their responses.
     *
     * @param request The combined request containing user prediction data and an order ID.
     * @return A CombinedPredictionResponse object.
     */
    @PostMapping
    public ResponseEntity<CombinedPredictionResponse> orchestratePrediction(@RequestBody CombinedPredictionRequest request) {
        System.out.println("Received orchestration request.");

        // 1. Get UserPredictionData from the request
        UserPredictionData userPredictionData = request.getUserPredictionData();
        Integer orderId = request.getOrderId();

        // Initialize variables for results
        OrderDetailsDTO orderDetails = null;
        Object predictionResult = null; // Use Object to be flexible with external API's JSON response

        // --- Debugging for userPredictionData ---
        if (userPredictionData == null) {
            System.out.println("DEBUG: userPredictionData object from request is NULL.");
        } else {
            System.out.println("DEBUG: userPredictionData object is NOT NULL.");
            if (userPredictionData.getUserDataList() == null) {
                System.out.println("DEBUG: userPredictionData.getUserDataList() is NULL.");
            } else {
                System.out.println("DEBUG: userPredictionData.getUserDataList() is NOT NULL. Size: " + userPredictionData.getUserDataList().size());
                System.out.println("DEBUG: userPredictionData.getUserDataList() content: " + userPredictionData.getUserDataList());
            }
        }
        // --- End Debugging ---


        // --- Call OrderService ---
        if (orderId != null) {
            System.out.println("Calling OrderService for order ID: " + orderId);
            try {
                // Make the call to OrderService to get order details with user info
                orderDetails = restTemplate.getForObject(
                        ORDER_SERVICE_WITH_USER_URL,
                        OrderDetailsDTO.class,
                        orderId // Path variable for the URL
                );
                System.out.println("Successfully fetched order details from OrderService.");
            } catch (HttpClientErrorException e) {
                System.err.println("Client error calling OrderService (status " + e.getStatusCode() + "): " + e.getResponseBodyAsString());
                return ResponseEntity.status(e.getStatusCode()).body(null);
            } catch (ResourceAccessException e) {
                System.err.println("Network/connection error calling OrderService: " + e.getMessage());
                return ResponseEntity.status(503).body(null);
            } catch (Exception e) {
                System.err.println("Unexpected error calling OrderService: " + e.getMessage());
                e.printStackTrace();
                return ResponseEntity.status(500).body(null);
            }
        } else {
            System.out.println("No orderId provided in the request. Skipping OrderService call.");
        }


        // --- Call External Prediction API ---
        if (userPredictionData != null && userPredictionData.getUserDataList() != null && !userPredictionData.getUserDataList().isEmpty()) {
            System.out.println("Calling external Prediction API.");
            try {
                // Create a PredictionApiRequest object, which wraps the user_data list
                PredictionApiRequest apiRequest = new PredictionApiRequest(userPredictionData.getUserDataList());

                // Manually serialize the object to a JSON string
                String jsonString = objectMapper.writeValueAsString(apiRequest);
                System.out.println("DEBUG: JSON sent to external API: " + jsonString);

                // Convert the JSON string to bytes
                byte[] requestBodyBytes = jsonString.getBytes("UTF-8");

                // Create HttpHeaders and set Content-Type and Content-Length
                HttpHeaders headers = new HttpHeaders();
                headers.setContentType(MediaType.APPLICATION_JSON);
                headers.setContentLength(requestBodyBytes.length); // Explicitly set Content-Length

                // Create an HttpEntity with the byte array body and headers
                HttpEntity<byte[]> requestEntity = new HttpEntity<>(requestBodyBytes, headers);

                // Use exchange method to send the request with explicit HttpEntity
                ResponseEntity<Object> responseEntity = restTemplate.exchange(
                        EXTERNAL_PREDICT_API_URL,
                        org.springframework.http.HttpMethod.POST, // Specify POST method
                        requestEntity, // Send the HttpEntity
                        Object.class // Expecting a generic JSON object/array from the external API
                );

                predictionResult = responseEntity.getBody();
                System.out.println("Successfully received response from external Prediction API.");
            } catch (HttpClientErrorException e) {
                System.err.println("Client error calling external Prediction API (status " + e.getStatusCode() + "): " + e.getResponseBodyAsString());
                return ResponseEntity.status(e.getStatusCode()).body(null);
            } catch (ResourceAccessException e) {
                System.err.println("Network/connection error calling external Prediction API: " + e.getMessage());
                return ResponseEntity.status(503).body(null);
            } catch (Exception e) {
                System.err.println("Unexpected error calling external Prediction API: " + e.getMessage());
                e.printStackTrace();
                return ResponseEntity.status(500).body(null);
            }
        } else {
            System.out.println("No userPredictionData or user_data list provided, or list is empty. Skipping external API call.");
        }


        // --- Combine Results and Return ---
        CombinedPredictionResponse response = new CombinedPredictionResponse(predictionResult, orderDetails);
        System.out.println("Orchestration complete. Returning combined response.");
        return ResponseEntity.ok(response);
    }
}