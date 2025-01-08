package hk.edu.polyu.comp.comp2021.cvfs.model;

import hk.edu.polyu.comp.comp2021.cvfs.exceptions.*;
import hk.edu.polyu.comp.comp2021.cvfs.model.criterion.Criterion;
import hk.edu.polyu.comp.comp2021.cvfs.model.criterion.IsDocCriterion;

import java.io.Serializable;
import java.util.HashMap;

/**
 * A virtual disk that contains directories and documents.
 * The virtual disk has a maximum size, and the size of the disk is the sum of the sizes of all directories and documents.
 */
public class VirtualDisk implements Serializable {
    private final Directory root;
    private final HashMap<String, Criterion> cMap;
    private final long maxSize;

    /**
     * <p>Creates a new virtual disk with the given maximum size.</p>
     * <p>The maximum size must be positive.</p>
     * @param maxSize the maximum size of the virtual disk
     */
    public VirtualDisk(long maxSize) {
        this.maxSize = maxSize;
        cMap = new HashMap<>();
        cMap.put("IsDocument", new IsDocCriterion());
        try {
            root = new Directory("root", null);
        } catch (InvalidFileException e) {
            throw new RuntimeException("unexpected during disk creation: " + e);
        }
    }

    /**
     * @return the root directory of the virtual disk
     */
    public Directory getRoot() { return root; }

    /**
     * @return a hash map of the criteria in the virtual disk in the form of <code>String crtName, Criterion crt</code>
     */
    public HashMap<String, Criterion> getCMap() { return cMap; }


    /**
     * @param crtName criterion name of the criterion to be added
     * @param crt criterion object to be added
     * @throws InvalidCriterionException when the criterion already exists
     */
    public void newCrt (String crtName, Criterion crt) throws InvalidCriterionException {
        assertCrtNotExist(crtName);
        cMap.put(crtName, crt);
    }

    /**
     * @param crtName criterion name of the criterion to be retrieved
     * @return the criterion object with the given name
     * @throws InvalidCriterionException when the criterion does not exist
     */
    public Criterion getCrt (String crtName) throws InvalidCriterionException{
        assertCrtExist(crtName);
        return cMap.get(crtName);
    }

    /**
     * @param crtName criterion name of the criterion to be deleted
     * @throws InvalidCriterionException when the criterion does not exist
     */
    public void delCrt(String crtName) throws InvalidCriterionException {
        assertCrtExist(crtName);
        cMap.remove(crtName);
    }


    /**
     * @param baseDir the directory to add the new document
     * @param doc the document object to be added
     * @throws FileExistsException when a document or directory with the same name already exists in the directory
     * @throws FullDiskException when the remaining space in the disk is insufficient for the new document
     */
    public void newDoc(Directory baseDir, Document doc) throws FileExistsException, FullDiskException {
        assertEnoughSpace(doc);
        baseDir.newDoc(doc);
    }

    /**
     * @param baseDir the directory to add the new directory
     * @param dir the directory object to be added
     * @throws FileExistsException when a document or directory with the same name already exists in the directory
     * @throws FullDiskException when the remaining space in the disk is insufficient for the new directory
     */
    public void newDir(Directory baseDir, Directory dir)
            throws FileExistsException, FullDiskException {
        assertEnoughSpace(dir);
        baseDir.newDir(dir);
    }

    private void assertEnoughSpace (Directory dir) throws FullDiskException{
        if (root.getSize() + dir.getSize() <= maxSize) return;
        String errorMessage = String.format(
                "out of disk space!\nVirtual disk: %d out of %d used\nSize required: %d\n",
                root.getSize(), maxSize, dir.getSize()
        );
        throw new FullDiskException(errorMessage);
    }

    private void assertEnoughSpace (Document doc) throws FullDiskException{
        if (root.getSize() + doc.getSize() <= maxSize) return;
        String errorMessage = String.format(
                "out of disk space!\nVirtual disk: %d out of %d used\nSize required: %d\n",
                root.getSize(), maxSize, doc.getSize()
        );
        throw new FullDiskException(errorMessage);
    }

    private void assertCrtExist (String crtName) throws InvalidCriterionException{
        if (cMap.containsKey(crtName)) return;
        throw Criterion.getInvalidCriterionException("criterion", "existing criterion", crtName);
    }

    private void assertCrtNotExist (String crtName) throws InvalidCriterionException {
        if (!cMap.containsKey(crtName)) return;
        throw Criterion.getInvalidCriterionException("criterion", "non-existing criterion", crtName);
    }
}
