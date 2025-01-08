package hk.edu.polyu.comp.comp2021.cvfs.model;

import hk.edu.polyu.comp.comp2021.cvfs.exceptions.InvalidFileException;
import hk.edu.polyu.comp.comp2021.cvfs.exceptions.InvalidTypeException;

import java.io.Serializable;

import static hk.edu.polyu.comp.comp2021.cvfs.utils.AssertUtils.assertValidFileName;

/**
 * <p>A document in the virtual disk.</p>
 */
public class Document implements Serializable {
    private static final long DOCUMENT_BASE_SIZE = 40;
    private String docName;
    private final FileType type;
    private final long size;
    private final Directory pDir; // Parent Directory

    // GETTERS, SETTERS

    /**
     * @return the size of the document
     */
    public long getSize() { return size; }

    /**
     * @return the fle type of the document
     */
    public FileType getType() { return type; }

    /**
     * @return the parent directory of the document
     */
    public Directory getPa() { return pDir; }

    /**
     * Rename the document.
     * @param docName the new name of the document
     */
    public void setDocName(String docName) { this.docName = docName; }

    /**
     * @return the name of the document
     */
    public String getDocName() { return docName; }

    // docName, type, content

    /**
     * <p>Creates a new document with the given name, content, and parent directory.</p>
     * <p>The name must be alphanumeric and ≤ 10 characters.</p>
     * <p>The size of the new document is 40 + 2 * content.length().</p>
     *
     * @param pDir the parent directory of the new document
     * @param docName the name of the new document
     * @param content the content of the new document
     * @param type the type of the new document
     * @throws InvalidTypeException when the type is invalid
     * @throws InvalidFileException when the name is invalid
     */
    public Document(Directory pDir, String docName, String content, String type) throws InvalidTypeException, InvalidFileException {
        assertValidFileName("document name", docName);
        this.docName = docName;
        this.type = FileType.getFileType(type);
        this.size = calcSize(content);
        this.pDir = pDir;
    }


    /**
     * @param content the content of the document
     * @return the calculated size of the document
     */
    static long calcSize (String content) {
        return DOCUMENT_BASE_SIZE + ((long) content.length() << 1);
    }
}