package utils;

import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import java.security.spec.InvalidKeySpecException;
import java.util.Base64;
import javax.crypto.SecretKeyFactory;
import javax.crypto.spec.PBEKeySpec;


public class SecurityUtils {
    // Character set for random BIN generation
    private static final SecureRandom RANDOM = new SecureRandom();
    // PBKDF2 configuration
    private static final String PBKDF2_ALGORITHM = "PBKDF2WithHmacSHA256";
    private static final int SALT_LENGTH = 16; // bytes
    private static final int HASH_LENGTH = 256; // bits
    private static final int ITERATIONS = 600_000;

    private static byte[] generateSalt() {
        byte[] salt = new byte[SALT_LENGTH];
        RANDOM.nextBytes(salt);
        return salt;
    }

    private static byte[] hashPassword(char[] password, byte[] salt) {
        PBEKeySpec spec = new PBEKeySpec(password, salt, ITERATIONS, HASH_LENGTH);
        try {
            SecretKeyFactory skf = SecretKeyFactory.getInstance(PBKDF2_ALGORITHM);
            return skf.generateSecret(spec).getEncoded();
        } catch (NoSuchAlgorithmException | InvalidKeySpecException e) {
            throw new RuntimeException("Error while hashing the password: " + e.getMessage(), e);
        } finally {
            spec.clearPassword();
        }
    }

    public static String toHash(char[] password) {
        byte[] salt = generateSalt();
        byte[] hash = hashPassword(password, salt);
        String encodedSalt = Base64.getEncoder().encodeToString(salt);
        String encodedHash = Base64.getEncoder().encodeToString(hash);
        return encodedSalt + ":" + encodedHash;
    }

    public static boolean checkPasswd(char[] password, String storedHash) {
        String[] parts = storedHash.split(":");
        assert parts.length == 2;
        byte[] salt = Base64.getDecoder().decode(parts[0]);
        byte[] hash = Base64.getDecoder().decode(parts[1]);
        byte[] computedHash = hashPassword(password, salt);
        return constTimeCmp(hash, computedHash);
    }

    /**
     * The password is valid only if the following conditions are met:
     * 1. The password contains 8 to 20 characters
     * 2. The password contains both uppercase and lowercase characters
     * 3. The password contains at least one digit
     * @param str the password to validate
     * @return true if the password is valid, false otherwise
     */
    public static boolean eval(char[] str) {
        if(str.length < 8 || str.length > 20) return false;
        boolean hasUpper = false, hasLower = false, hasDigit = false;
        for (char c : str) {
            if (Character.isUpperCase(c)) hasUpper = true;
            if (Character.isLowerCase(c)) hasLower = true;
            if (Character.isDigit(c)) hasDigit = true;
            if (c == ' ') return false;
        }
        return hasUpper && hasLower && hasDigit;
    }

    public static String reason() {
        return "Password must contain 8-20 characters, uppercase, lowercase, digit";
    }

    /**
     * Compares two byte arrays in constant amount of time to prevent timing attacks.
     * @param a First byte array.
     * @param b Second byte array.
     * @return True if both byte arrays are equal, false otherwise.
     */
    private static boolean constTimeCmp(byte[] a, byte[] b) {
        int diff = a.length ^ b.length;
        for (int i = 0; i < Math.min(a.length, b.length); i++) {
            diff |= a[i] ^ b[i];
        }
        return diff == 0;
    }
}