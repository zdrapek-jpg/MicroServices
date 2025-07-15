package org.example;


/**
 * DTO to mirror the User object from UserService, used for deserialization
 * within PredictionService.
 */
public class UserDetailsDTO {
    private Integer id;
    private String name;

    // Constructors
    public UserDetailsDTO() {
    }

    public UserDetailsDTO(Integer id, String name) {
        this.id = id;
        this.name = name;
    }

    // Getters and Setters
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