package hk.edu.polyu.comp.comp2021.cvfs.model;

import hk.edu.polyu.comp.comp2021.cvfs.exceptions.InvalidTypeException;

/**
 * <p>Represents the type of a file.</p>
 */
public enum FileType {
    /**
     * Cascading Style Sheets
     */
    CSS,

    /**
     * HyperText Markup Language
     */
    HTML,

    /**
     * Java source code
     */
    JAVA,

    /**
     * Plain text
     */
    TXT;


    /**
     * <p>Converts a string to a FileType object.</p>
     * <p>The string must be one of the following: {txt, java, html, css}</p>
     *
     * @param type the string to be converted
     * @return the FileType object corresponding to the string
     * @throws InvalidTypeException when the string is not one of the following: {txt, java, html, css}
     */
    public static FileType getFileType(String type) throws InvalidTypeException{
        switch (type) {
            case "txt": return FileType.TXT;
            case "java": return FileType.JAVA;
            case "html": return FileType.HTML;
            case "css": return FileType.CSS;
            default:
                throw new InvalidTypeException("expects file type of {txt, java, html, css} in lowercase, but got " + type);
        }
    }

    /**
     * @return the string representation of the FileType object in lowercase
     */
    public String toStrType () {
        return this.toString().toLowerCase();
    }
}
