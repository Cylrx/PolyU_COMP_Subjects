package utils.validator;

import java.time.format.DateTimeParseException;

public class TimeValidator implements Validator {
    @Override
    public boolean eval(String str) {
        if (str == null) return false;
        try {
            String[] strs = str.split(":");
            int[] parts = new int[strs.length];
            for (int i = 0; i < strs.length; i++) {
                if(strs[i].length() != 2) return false;
                try{
                    parts[i] = Integer.parseInt(strs[i]);
                }
                catch (NumberFormatException e) {
                    return false;
                }
            }
            return parts.length == 3 && parts[0] >= 0 && parts[0] <= 23 && parts[1] >= 0 && parts[1] <= 59 && parts[2] >= 0 && parts[2] <= 59;
        } catch (DateTimeParseException e) {
            return false;
        }
    }

    @Override
    public String reason() {
        return "Time must be in the format of HH:mm:ss (24-hour format)";
    }
}