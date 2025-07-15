package org.example;


/**
 * Data Transfer Object (DTO) representing the input structure for the PredictionService's
 * /predict endpoint. This object holds the individual fields that will be
 * assembled into the 'user_data' array for the external prediction API.
 */

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

/**
 * Data Transfer Object (DTO) representing the 'user_data' array
 * directly as expected by the external prediction API.
 * This class now holds a single list of objects.
 */
public class UserPredictionData {

    // Use @JsonProperty to map the JSON key "user_data" to this field.
    @JsonProperty("user_data")
    private List<Object> userDataList;

    // Constructors
    public UserPredictionData() {
    }

    public UserPredictionData(List<Object> userDataList) {
        this.userDataList = userDataList;
    }

    // Getter and Setter for the user_data list
    public List<Object> getUserDataList() {
        return userDataList;
    }

    public void setUserDataList(List<Object> userDataList) {
        this.userDataList = userDataList;
    }
}
