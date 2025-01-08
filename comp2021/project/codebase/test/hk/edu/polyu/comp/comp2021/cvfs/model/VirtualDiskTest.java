package hk.edu.polyu.comp.comp2021.cvfs.model;
import hk.edu.polyu.comp.comp2021.cvfs.exceptions.*;
import hk.edu.polyu.comp.comp2021.cvfs.model.criterion.*;
import org.junit.*;
import java.util.Random;

import static hk.edu.polyu.comp.comp2021.cvfs.model.TestUtils.getRandStr;
import static org.junit.Assert.*;

public class VirtualDiskTest {
    private VirtualDisk vd, bigVd;

    @Before
    public void setUp() throws InvalidFileException {
        vd = new VirtualDisk(1000);
        bigVd = new VirtualDisk(100000000);
    }

    // Static Tests
    @Test
    public void testValidVirtualDiskCreation() {
        assertNotNull(vd.getRoot());
        assertEquals(1, vd.getCMap().size());
        assertTrue(vd.getCMap().containsKey("IsDocument"));
        assertEquals("root", vd.getRoot().getDirName());
        assertNull(vd.getRoot().getPa());
    }

    @Test
    public void testNewCriterion() throws Exception {
        Criterion c = new NameCriterion("NameCrt", "contains", "\"test\"");
        vd.newCrt("NameCrt", c);
        assertTrue(vd.getCMap().containsKey("NameCrt"));
        assertEquals(c, vd.getCMap().get("NameCrt"));
        assertEquals(c, vd.getCrt("NameCrt"));
    }

    @Test
    public void testNewCriterionAlreadyExists() throws Exception {
        Criterion c = new NameCriterion("NameCrt", "contains", "\"test\"");
        vd.newCrt("NameCrt", c);
        try {
            vd.newCrt("NameCrt", c);
            fail("Expected InvalidCriterionException");
        } catch (InvalidCriterionException e) {
            //
        }
    }

    @Test
    public void testDeleteCriterion() throws Exception {
        Criterion c = new SizeCriterion("SizeCrt", ">", "100");
        vd.newCrt("SizeCrt", c);
        vd.delCrt("SizeCrt");
        assertFalse(vd.getCMap().containsKey("SizeCrt"));
    }

    @Test
    public void testDeleteNonExistentCriterion() {
        try {
            vd.delCrt("NoCrt");
            fail("Expected InvalidCriterionException");
        } catch (InvalidCriterionException e) {
            // skip
        }
    }

    @Test
    public void testNewDocumentWithinSpace() throws Exception {
        Directory base = vd.getRoot();
        Document doc = new Document(base, "Doc1", "Content", "txt");
        vd.newDoc(base, doc);
        assertTrue(base.getSubDocs().containsKey("Doc1"));
        assertEquals(1, base.getCnt());
        assertEquals(doc.getSize(), base.getSize());
    }

    @Test
    public void testNewBigDoc() throws Exception {
        Random rand = new Random();
        // Tries to add document that's too big
        Directory base = vd.getRoot();
        Document doc = new Document(base, "BigDoc", getRandStr(rand, 500), "java");
        try {
            vd.newDoc(base, doc);
            fail("Expected FullDiskException");
        } catch (FullDiskException e) {
            assertTrue(e.getMessage().contains("out of disk space"));
        }
    }

    @Test
    public void testNewDirWithinSpace() throws Exception {
        Directory base = vd.getRoot();
        Directory dir = new Directory("Dir1", base);
        vd.newDir(base, dir);
        assertTrue(base.getSubDirs().containsKey("Dir1"));
        assertEquals(dir.getSize(), base.getSize());
    }

    @Test
    public void testNewBigDir() throws Exception {
        // Tries to add directory that's too big
        Directory base = vd.getRoot();
        Directory dir = new Directory("BigDir", base);
        for(int i = 0; i < 3000; i++) {
            Document doc = new Document(dir, "D" + i, "Data", "txt");
            dir.newDoc(doc);
        }
        try {
            vd.newDir(base, dir);
            fail("Expected FullDiskException");
        } catch (FullDiskException e) {
            assertTrue(e.getMessage().contains("out of disk space"));
        }
    }

    // Enhanced Fuzz Tests
    @Test
    public void fuzzTestVirtualDiskOperations() {
        Random rand = new Random();
        String[] validTypes = {"txt", "java", "html", "css"};
        String[] invalidTypes = {"exe", "pdf", "lol", " ", "   "};

        // valid file types
        for(int i = 0; i < 50; i++) {
            String docName = getRandStr(rand, rand.nextInt(10) + 1);
            String content = getRandStr(rand, rand.nextInt(50) + 1);
            String type = validTypes[rand.nextInt(validTypes.length)];
            try {
                Directory base = bigVd.getRoot();
                Document doc = new Document(base, docName, content, type);
                bigVd.newDoc(base, doc);
                assertTrue(base.getSubDocs().containsKey(docName));
            } catch (InvalidFileException | InvalidTypeException e) {
                fail("Unexpected exception with valid input: " + e.getMessage());
            } catch (FullDiskException | FileExistsException e) {
                // possible, skip
            }
        }

        // Testing with invalid file types
        for(int i = 0; i < 50; i++) {
            String docName = getRandStr(rand, rand.nextInt(10) + 1);
            String content = getRandStr(rand, rand.nextInt(50) + 1);
            String type = invalidTypes[rand.nextInt(invalidTypes.length)];
            try {
                Directory base = bigVd.getRoot();
                Document doc = new Document(base, docName, content, type);
                bigVd.newDoc(base, doc);
                fail("Expected InvalidTypeException for type: " + type);
            } catch (InvalidFileException e) {
                fail("Unexpected InvalidFileException: " + e.getMessage());
            } catch (InvalidTypeException e) {
                assertTrue(e.getMessage().contains("expects file type"));
            } catch (FullDiskException | FileExistsException e) {
                // possible, skip
            }
        }
    }
}