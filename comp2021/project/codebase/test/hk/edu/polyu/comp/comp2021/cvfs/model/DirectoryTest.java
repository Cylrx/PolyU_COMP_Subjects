package hk.edu.polyu.comp.comp2021.cvfs.model;

import hk.edu.polyu.comp.comp2021.cvfs.exceptions.*;
        import org.junit.*;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;

import static hk.edu.polyu.comp.comp2021.cvfs.model.TestUtils.*;
import static org.junit.Assert.*;

/**
 *
 */
public class DirectoryTest {
    private Directory root;

    @Before
    public void setUp() throws InvalidFileException {
        root = new Directory("root", null);
        assertEquals(0, root.getSize());
        assertEquals(0, root.getCnt());
    }

    // Static Tests

    @Test
    public void testValidDirectoryCreation() throws InvalidFileException {
        Directory dir = new Directory("Folder1", root);
        assertEquals("Folder1", dir.getDirName());
        assertEquals(root, dir.getPa());
        assertEquals(40, dir.getSize());
        assertEquals(1, dir.getCnt());
    }

    @Test
    public void testInvalidDirectoryNameNonAlphanumeric() {
        try {
            new Directory("Folder@", root);
            fail("Expected InvalidFileException");
        } catch (InvalidFileException e) {
            assertEquals("invalid directory name: expects alphanumeric, length <= 10", e.getMessage());
        }
    }

    @Test
    public void testInvalidDirectoryNameTooLong() {
        try {
            new Directory("LongFolderNameExceed", root);
            fail("Expected InvalidFileException");
        } catch (InvalidFileException e) {
            assertEquals("invalid directory name: expects alphanumeric, length <= 10", e.getMessage());
        }
    }

    @Test
    public void testNewDocument() throws Exception {
        Document doc = new Document(root, "Doc1", "Content", "txt");
        root.newDoc(doc);
        assertTrue(root.getSubDocs().containsKey("Doc1"));
        assertEquals(doc, root.getSubDocs().get("Doc1"));
        assertEquals(doc.getSize(), root.getSize());
        assertEquals(1, root.getCnt());
    }

    @Test
    public void testNewDirectory() throws Exception {
        Directory dir = new Directory("SubDir", root);
        root.newDir(dir);
        assertTrue(root.getSubDirs().containsKey("SubDir"));
        assertEquals(dir, root.getSubDirs().get("SubDir"));
        assertEquals(dir.getSize(), root.getSize());
    }

    @Test
    public void testDeleteDocument() throws Exception {
        Document doc = new Document(root, "Doc1", "Content", "java");
        root.newDoc(doc);
        Object deleted = root.delete("Doc1");
        assertEquals(doc, deleted);
        assertFalse(root.getSubDocs().containsKey("Doc1"));
        assertEquals(0, root.getCnt());
        assertEquals(0, root.getSize());
    }

    @Test
    public void testDeleteDirectory() throws Exception {
        Directory subDir = new Directory("SubDir", root);
        root.newDir(subDir);
        Object deleted = root.delete("SubDir");
        assertEquals(subDir, deleted);
        assertFalse(root.getSubDirs().containsKey("SubDir"));
        assertEquals(0, root.getSize());
    }

    @Test
    public void testDeleteNonExistent() {
        try {
            root.delete("NoExist");
            fail("Expected FileNotFoundException");
        } catch (FileNotFoundException e) {
            assertEquals("cannot find file: \"NoExist\"", e.getMessage());
        } catch (Exception e){
            fail("Unexpected exception");
        }
    }

    @Test
    public void testRenameDocument() throws Exception {
        Document doc = new Document(root, "Doc1", "Content", "css");
        root.newDoc(doc);
        root.rename("Doc1", "DocRenamed");
        assertFalse(root.getSubDocs().containsKey("Doc1"));
        assertTrue(root.getSubDocs().containsKey("DocRenamed"));
        assertEquals("DocRenamed", root.getSubDocs().get("DocRenamed").getDocName());
    }

    @Test
    public void testRenameDirectory() throws Exception {
        Directory dir = new Directory("SubDir", root);
        root.newDir(dir);
        root.rename("SubDir", "SubRenamed");
        assertFalse(root.getSubDirs().containsKey("SubDir"));
        assertTrue(root.getSubDirs().containsKey("SubRenamed"));
        assertEquals("SubRenamed", root.getSubDirs().get("SubRenamed").getDirName());
    }

    @Test
    public void testRenameToExistingName() throws Exception {
        Directory dir1 = new Directory("Dir1", root);
        Directory dir2 = new Directory("Dir2", root);
        root.newDir(dir1);
        root.newDir(dir2);
        try {
            root.rename("Dir1", "Dir2");
            fail("Expected FileExistsException");
        } catch (FileExistsException e) {
            assertEquals("file \"Dir2\" already exists as a directory", e.getMessage());
        }
    }

    // Fuzz Tests
    @Test
    public void fuzzTestRandDocOperation() throws InvalidFileException, InvalidTypeException, FileExistsException {
        Random rand = new Random();
        HashSet<String> exist = new HashSet<>();
        Directory p = new Directory("p", null);

        for (int i = 0; i < 1000; i++) {
            String name = getRandStr(rand, 2);
            while(exist.contains(name)) name = getRandStr(rand, 2);
            exist.add(name);
            p.newDoc(new Document(p, name, "", getRandFileType(rand)));
        }

        while (!exist.isEmpty()) {
            String oldName = getRandStr(rand, 2);
            String newName = getRandStr(rand, rand.nextInt(4) + 8);
            boolean contains = exist.contains(oldName);
            boolean validLength = newName.length() <= 10;

            try {
                Document doc = p.getDoc(oldName);
                if (!contains) fail("Expected FileNotFoundException: file" + oldName + " not exists");
                else {
                    assertNotNull(doc);
                    assertEquals(doc.getDocName(), oldName);
                }
            } catch (FileNotFoundException e) {
                if (contains) fail("Unexpected FileNotFoundException: file " + oldName + " exists");
                else assertEquals(e.getMessage(), "cannot find document: \"" + oldName + "\"");
            }

            try {
                p.rename(oldName, newName);
                if (!contains || !validLength) fail("Expected FileNotFoundException: file" + oldName + " not exists");
                else {
                    exist.remove(oldName);
                    assertTrue(p.getSubDocs().containsKey(newName));
                    assertFalse(p.getSubDocs().containsKey(oldName));
                }
            } catch (FileNotFoundException e) {
                if (contains) fail("Unexpected FileNotFoundException: file " + oldName + " exists");
                else assertEquals(e.getMessage(), "cannot find file: \"" + oldName + "\"");
            } catch (InvalidFileException e) {
                if (validLength) fail("Unexpected InvalidFileException: file name is valid");
            }
        }
    }

    @Test
    public void fuzzTestRandDirOperation() throws InvalidFileException, FileExistsException {
        Random rand = new Random();
        HashSet<String> exist = new HashSet<>();
        Directory p = new Directory("p", null);

        for (int i = 0; i < 1000; i++) {
            String name = getRandStr(rand, 2);
            while(exist.contains(name)) name = getRandStr(rand, 2);
            exist.add(name);
            p.newDir(new Directory(name, p));
        }

        while (!exist.isEmpty()) {
            String oldName = getRandStr(rand, 2);
            String newName = getRandStr(rand, 5);
            boolean contains = exist.contains(oldName);

            try {
                Directory dir = p.getDir(oldName);
                if (!contains) fail("Expected FileNotFoundException: file" + oldName + " not exists");
                else {
                    assertNotNull(dir);
                    assertEquals(dir.getDirName(), oldName);
                }
            } catch (FileNotFoundException e) {
                if (contains) fail("Unexpected FileNotFoundException: file " + oldName + " exists");
                else assertEquals( "cannot find directory: \"" + oldName + "\"", e.getMessage());
            }

            try {
                p.rename(oldName, newName);
                if (!contains) fail("Expected FileNotFoundException: file" + oldName + " not exists");
                else {
                    exist.remove(oldName);
                    assertTrue(p.getSubDirs().containsKey(newName));
                    assertFalse(p.getSubDirs().containsKey(oldName));
                }
            } catch (FileNotFoundException e) {
                if (contains) fail("Unexpected FileNotFoundException: file " + oldName + " exists");
                else assertEquals(e.getMessage(), "cannot find file: \"" + oldName + "\"");
            }
        }

    }

    @Test
    public void fuzzTestDirectoryOperations() {
        Random rand = new Random();
        try {
            for(int i = 0; i < 50; i++) {
                String dirName = getRandStr(rand, rand.nextInt(12));
                if(dirName.length() > 10 || !dirName.matches("[A-Za-z0-9]+")) continue;
                Directory dir = new Directory(dirName, root);
                root.newDir(dir);
                assertTrue(root.getSubDirs().containsKey(dirName));

                // Random rename
                String newName = getRandStr(rand, rand.nextInt(12));
                if(newName.length() > 10 || !newName.matches("[A-Za-z0-9]+") || root.getSubDirs().containsKey(newName))
                    continue;
                root.rename(dirName, newName);
                assertFalse(root.getSubDirs().containsKey(dirName));
                assertTrue(root.getSubDirs().containsKey(newName));
            }
        } catch (Exception e) {
            // ignored
        }
    }

    @Test
    public void fuzzTestRandomFileTreeConstruction() {
        Random rand = new Random();
        int operations = 20000;
        long expectedSize = root.getSize();
        long expectedCnt = root.getCnt();

        for(int i = 0; i < operations; i++) {
            boolean create = (rand.nextDouble() > 0.1);
            if(create) {
                boolean createDir = rand.nextBoolean();
                Directory targetDir = getRandDirectory(root, rand, 5);
                if(targetDir == null) continue;
                if(createDir) {
                    String dirName = getRandStr(rand, rand.nextInt(8)+1);
                    try {
                        Directory newDir = new Directory(dirName, targetDir);
                        targetDir.newDir(newDir);
                        expectedSize += newDir.getSize();
                        expectedCnt += 1;
                    } catch (InvalidFileException | FileExistsException e){
                        // skip
                    }
                } else {
                    String docName = getRandStr(rand, rand.nextInt(8)+1);
                    String content = getRandStr(rand, rand.nextInt(100));
                    String type = getRandFileType(rand);
                    if(notValidFileType(type)) continue;
                    try {
                        Document doc = new Document(targetDir, docName, content, type);
                        targetDir.newDoc(doc);
                        expectedSize += doc.getSize();
                        expectedCnt +=1;
                    } catch (InvalidFileException | InvalidTypeException | FileExistsException e){
                        // skip
                    }
                }
            } else {
                Directory targetDir = getRandDirectory(root, rand, 5);
                if(targetDir == null) continue;
                List<String> keys = new ArrayList<>();
                keys.addAll(targetDir.getSubDirs().keySet());
                keys.addAll(targetDir.getSubDocs().keySet());
                if(keys.isEmpty()) continue;
                String key = keys.get(rand.nextInt(keys.size()));
                try {
                    Object deleted = targetDir.delete(key);
                    if (deleted instanceof Document){
                        Document doc = (Document) deleted;
                        expectedSize -= doc.getSize();
                        expectedCnt -=1;
                    } else if(deleted instanceof Directory){
                        Directory dir = (Directory) deleted;
                        expectedSize -= dir.getSize();
                        expectedCnt -= dir.getCnt();
                    }
                } catch (FileNotFoundException e){
                    // skip
                }
            }
            assertEquals(expectedSize, root.getSize());
            assertEquals(expectedCnt, root.getCnt());
        }
    }

    private long countDocs(Directory dir) {
        long count = dir.getSubDocs().size();
        for(Directory subDir : dir.getSubDirs().values()) {
            count += countDocs(subDir);
        }
        return count;
    }

    private Directory getRandDirectory(Directory current, Random rand, int maxDepth) {
        if(maxDepth ==0) return current;
        ArrayList<Directory> dirs = new ArrayList<>(current.getSubDirs().values());
        if(dirs.isEmpty()) return current;
        boolean goDeeper = rand.nextBoolean();
        if(goDeeper) {
            Directory nextDir = dirs.get(rand.nextInt(dirs.size()));
            return getRandDirectory(nextDir, rand, maxDepth-1);
        } else {
            return current;
        }
    }
}