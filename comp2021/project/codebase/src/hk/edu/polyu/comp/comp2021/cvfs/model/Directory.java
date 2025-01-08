package hk.edu.polyu.comp.comp2021.cvfs.model;

import hk.edu.polyu.comp.comp2021.cvfs.exceptions.FileExistsException;
import hk.edu.polyu.comp.comp2021.cvfs.exceptions.FileNotFoundException;
import hk.edu.polyu.comp.comp2021.cvfs.exceptions.InvalidFileException;

import java.io.Serializable;
import java.util.HashMap;

import static hk.edu.polyu.comp.comp2021.cvfs.utils.AssertUtils.assertValidFileName;

/**
 * <p>A directory in the virtual disk.</p>
 */
public class Directory implements Serializable {
    private static final long DIRECTORY_BASE_SIZE = 40L;
    private final HashMap<String, Directory> dirMap;
    private final HashMap<String, Document> docMap;
    private final Directory pDir; // parent Directory
    private String dirName;
    private long size;
    private long cnt;

    /**
     * <p>Creates a new directory with the given name and parent directory.</p>
     * <p>The name must be alphanumeric and ≤ 10 characters.</p>
     * <p>The parent directory can be null, which means the new directory is the root directory.</p>
     * <p>The size of the new directory is 0 if the parent directory is null, otherwise 40.</p>
     * <p>The count of the new directory is 0.</p>
     *
     * @param dirName the name of the new directory
     * @param pDir the parent directory of the new directory
     * @throws InvalidFileException when the name is invalid
     */
    public Directory (String dirName, Directory pDir) throws InvalidFileException{
        assertValidFileName("directory name", dirName);
        dirMap = new HashMap<>();
        docMap = new HashMap<>();
        this.dirName = dirName;
        this.pDir = pDir;
        size = (pDir == null) ? 0 : DIRECTORY_BASE_SIZE;
        cnt = (pDir == null) ? 0: 1;
    }

    private void updateSize(long delta) {
        size += delta;
        if (pDir != null) pDir.updateSize(delta);
    }

    private void updateCnt(long delta) {
        cnt += delta;
        if (pDir != null) pDir.updateCnt(delta);
    }

    // GETTERS, SETTERS

    /**
     * @return the number of files in the directory
     */
    public long getCnt () { return cnt; }

    /**
     * @return the size of the directory, including all files within
     */
    public long getSize () { return size; }

    /**
     * @return the name of the directory
     */
    public String getDirName() { return dirName; }

    /**
     * Renames the current directory
     * @param dirName the new name of the directory
     */
    public void setDirName(String dirName) { this.dirName = dirName; }

    /**
     * @return the parent directory of the current directory
     */
    public Directory getPa () { return pDir; }

    /**
     * @return a hash map of all the subdirectories in the form of <code>String name, Directory dir</code>
     */
    public HashMap<String, Directory> getSubDirs() { return dirMap; }

    /**
     * @return a hash map of all the documents in the form of <code>String name, Document doc</code>
     */
    public HashMap<String, Document> getSubDocs() { return docMap; }


    // GET, SET, NEW, DEL commands

    // newDoc and newDir are protected to restrict their use outside "Model".
    // This ensures Doc and Dir are created via VirtualDisk,
    // which checks for space availability (assertEnoughSpace).

    /**
     * <p>Adds a new document to the directory.</p>
     * <p>The document must have a unique name within the directory.</p>
     * <p>The size and count of the directory are updated accordingly.</p>
     *
     * @param doc the document to be added
     * @throws FileExistsException when a document with the same name already exists in the directory
     */
    protected void newDoc (Document doc) throws FileExistsException{
        String key = doc.getDocName();
        assertUniqueName(key);
        docMap.put(key, doc);
        updateSize(doc.getSize());
        updateCnt(1);
    }

    /**
     * <p>Adds a new directory to the directory.</p>
     * <p>The directory must have a unique name within the directory.</p>
     * <p>The size and count of the directory are updated accordingly.</p>
     *
     * @param dir the directory to be added
     * @throws FileExistsException when a directory with the same name already exists in the directory
     */
    protected void newDir (Directory dir) throws FileExistsException{
        String key = dir.getDirName();
        assertUniqueName(key);
        dirMap.put(key, dir);
        updateSize(dir.getSize());
        updateCnt(1);
    }

    private void delDoc (String key) {
        Document doc = docMap.get(key);
        docMap.remove(key);
        updateSize(-doc.getSize());
        updateCnt(-1);
    }

    private void delDir (String key) {
        Directory dir = dirMap.get(key);
        dirMap.remove(key);
        updateSize(-dir.getSize());
        updateCnt(-dir.getCnt());
    }

    /**
     * <p>Deletes a file from the directory.</p>
     * <p>The file can be either a document or a directory.</p>
     * <p>The size and count of the directory are updated accordingly.</p>
     *
     * @param key the name of the file to be deleted
     * @return the file that was deleted
     * @throws FileNotFoundException when the file does not exist in the directory
     */
    public Object delete (String key) throws FileNotFoundException {
        if (dirMap.containsKey(key)) {
            Directory dir = getDir(key);
            delDir(key);
            return dir;
        }
        if (docMap.containsKey(key)) {
            Document doc = getDoc(key);
            delDoc(key);
            return doc;
        }
        throw new FileNotFoundException("cannot find file: \"" + key + "\"");
    }

    /**
     * <p>Renames a file in the directory.</p>
     * <p>The file can be either a document or a directory.</p>
     * <p>The size and count of the directory are updated accordingly.</p>
     *
     * @param oldName the current name of the file
     * @param newName the new name of the file
     * @throws FileExistsException when a file with the new name already exists in the directory
     * @throws FileNotFoundException when the file with the old name does not exist in the directory
     * @throws InvalidFileException when the new file name is invalid
     */
    public void rename(String oldName, String newName) throws FileExistsException, FileNotFoundException, InvalidFileException {
        assertValidFileName("new file name", newName);
        if (dirMap.containsKey(oldName)) {
            setSubDirName(oldName, newName);
            return;
        }
        if (docMap.containsKey(oldName)) {
            setSubDocName(oldName, newName);
            return;
        }
        throw new FileNotFoundException("cannot find file: \"" + oldName + "\"");
    }

    /**
     * <p>Gets a subdirectory from the directory.</p>
     * <p>The subdirectory must exist in the directory.</p>
     *
     * @param key the name of the subdirectory
     * @return the subdirectory
     * @throws FileNotFoundException when the subdirectory does not exist in the directory
     */

    public Directory getDir (String key) throws FileNotFoundException {
        assertDirExists(key);
        return dirMap.get(key);
    }

    /**
     * <p>Gets a document from the directory.</p>
     * <p>The document must exist in the directory.</p>
     *
     * @param key the name of the document
     * @return the document
     * @throws FileNotFoundException when the document does not exist in the directory
     */
    public Document getDoc (String key) throws FileNotFoundException{
        assertDocExists(key);
        return docMap.get(key);
    }

    private void setSubDirName (String oldDirName, String newDirName) throws FileNotFoundException, FileExistsException{
        assertDirExists(oldDirName);
        assertUniqueName(newDirName);
        Directory dir = dirMap.get(oldDirName);
        dir.setDirName(newDirName);
        dirMap.remove(oldDirName);
        dirMap.put(newDirName, dir);
    }

    private void setSubDocName (String oldDocName, String newDocName) throws FileNotFoundException, FileExistsException{
        assertDocExists(oldDocName);
        assertUniqueName(newDocName);
        Document doc = docMap.get(oldDocName);
        doc.setDocName(newDocName);
        docMap.remove(oldDocName);
        docMap.put(newDocName, doc);
    }



    // ASSERTIONS

    private void assertDirExists(String key) throws FileNotFoundException {
        if (dirMap.containsKey(key)) return;
        throw new FileNotFoundException("cannot find directory: \"" + key + "\"");
    }

    private void assertDocExists(String key) throws FileNotFoundException {
        if (docMap.containsKey(key)) return;
        throw new FileNotFoundException("cannot find document: \"" + key + "\"");
    }

    private void assertUniqueName(String key) throws FileExistsException {
        if (docMap.containsKey(key)) throw new FileExistsException("file \"" + key + "\" already exists as a document");
        if (dirMap.containsKey(key)) throw new FileExistsException("file \"" + key + "\" already exists as a directory");
    }
}
