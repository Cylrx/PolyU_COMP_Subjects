package hk.edu.polyu.comp.comp2021.cvfs.utils;

import hk.edu.polyu.comp.comp2021.cvfs.exceptions.InvalidIOException;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * A utility class for serializing and deserializing objects.
 */
public class SerializeUtils {

    /**
     * Serializes an object to a file.
     * @param obj the object to be serialized
     * @param pathStr the path of the file to be written
     * @throws InvalidIOException when the path is invalid or the file cannot be written
     */
    public static void serialize(Object obj, String pathStr) throws InvalidIOException {
        Path path = getWritablePath(pathStr);
        try (FileOutputStream fos = new FileOutputStream(path.toFile());
             ObjectOutputStream oos = new ObjectOutputStream(fos)) {
            oos.writeObject(obj);
        } catch (IOException e) {
            throw new InvalidIOException("Internal IOException: " + e);
        }
    }

    /**
     * Deserializes an object from a file.
     * @param pathStr the path of the file to be read
     * @return the deserialized object
     * @throws InvalidIOException when the path is invalid or the file cannot be read
     */
    public static Object deserialize (String pathStr) throws InvalidIOException{
        Path path = getReadablePath(pathStr);
        try (FileInputStream fis = new FileInputStream(path.toFile());
             ObjectInputStream ois = new ObjectInputStream(fis)) {
            return ois.readObject();
        } catch (IOException e) {
            throw new InvalidIOException("internal IOException: " + e.getMessage());
        } catch (ClassNotFoundException e) {
            throw new InvalidIOException("class not found exception" + e.getMessage());
        }
    }

    /**
     * Performs various checks to see if a file can be written to.
     * @param pathStr the path of the file to be written to
     * @return the Path object of the file
     * @throws InvalidIOException when the path is invalid or the file cannot be written due to various reasons
     */
    public static Path getWritablePath(String pathStr) throws InvalidIOException {
        if (pathStr == null || pathStr.trim().isEmpty())
            throw new InvalidIOException("Cannot write file: null or empty path");
        Path path;
        try {
            path = Paths.get(pathStr);
        } catch (InvalidPathException e) {
            throw new InvalidIOException("Cannot write file: invalid path format " + pathStr);
        }
        Path parent = path.getParent();
        if (parent != null) {
            if (!Files.exists(parent))
                throw new InvalidIOException("Cannot write file: parent directory does not exist");
            if (!Files.isWritable(parent))
                throw new InvalidIOException("Cannot write file: no permission for parent directory" + parent);
        }
        if (Files.exists(path)) {
            if (Files.isDirectory(path))
                throw new InvalidIOException("Cannot write file: path points to a directory, not a file");
            else
                throw new InvalidIOException("Cannot write file: file already exists");
        }
        return path;
    }

    /**
     * Performs various checks to see if a file can be read.
     * @param pathStr the path of the file to be read
     * @return the Path object of the file
     * @throws InvalidIOException when the path is invalid or the file cannot be read due to various reasons
     */
    public static Path getReadablePath(String pathStr) throws InvalidIOException {
        if (pathStr == null || pathStr.trim().isEmpty())
            throw new InvalidIOException("Cannot read file: null or empty path");
        Path path;
        try {
            path = Paths.get(pathStr);
        } catch (InvalidPathException e) {
            throw new InvalidIOException("Cannot read file: invalid path format " + pathStr);
        }
        if (!Files.exists(path))
            throw new InvalidIOException("Cannot read file: file not found");
        if (Files.isDirectory(path))
            throw new InvalidIOException("Cannot read file: path points to a directory, not a file");
        if (!Files.isReadable(path))
            throw new InvalidIOException("Cannot read file: no permission for file " + pathStr);
        return path;
    }
}
