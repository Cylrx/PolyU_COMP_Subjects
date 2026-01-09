package model.logging;

public enum EventType {
    Captured("Captured"), Move("Move"), Withdraw("Withdraw");

    private final String name;
    private EventType(String name) {
        this.name = name;
    }
    public String toString() {
        return name;
    }
}
