package hk.edu.polyu.comp.comp2021.cvfs.model;

import hk.edu.polyu.comp.comp2021.cvfs.exceptions.InvalidFileException;
import hk.edu.polyu.comp.comp2021.cvfs.exceptions.InvalidTypeException;
import org.junit.*;
import java.util.Random;

import static hk.edu.polyu.comp.comp2021.cvfs.model.TestUtils.*;
import static org.junit.Assert.*;

public class DocumentTest {
    private Directory dummyDir;

    @Before
    public void setUp() throws InvalidFileException {
        dummyDir = new Directory("rootDir", null);
    }

    // Static Tests
    @Test
    public void testValidDocumentCreation() throws InvalidTypeException, InvalidFileException {
        Document doc = new Document(dummyDir, "Doc1", "Hello World", "txt");
        assertEquals("Doc1", doc.getDocName());
        assertEquals(FileType.TXT, doc.getType());
        assertEquals(40 + (11 << 1), doc.getSize());
        assertEquals(dummyDir, doc.getPa());
    }

    @Test
    public void testCalcSize() {
        assertEquals(40, Document.calcSize(""));
        assertEquals(42, Document.calcSize("a"));
        assertEquals(60, Document.calcSize("abcdefghij"));
    }

    @Test
    public void testSetDocName() throws InvalidTypeException, InvalidFileException {
        Document doc = new Document(dummyDir, "Doc1", "Content", "java");
        doc.setDocName("NewName");
        assertEquals("NewName", doc.getDocName());
    }

    @Test
    public void testInvalidDocNameNonAlphanumeric() {
        try {
            new Document(dummyDir, "Doc@123", "Content", "html");
            fail("Expected InvalidFileException");
        } catch (InvalidFileException | InvalidTypeException e) {
            assertEquals("invalid document name: expects alphanumeric, length <= 10", e.getMessage());
        }
    }

    @Test
    public void testInvalidDocNameTooLong() {
        try {
            new Document(dummyDir, "DocumentNameExceed", "Content", "css");
            fail("Expected InvalidFileException");
        } catch (InvalidFileException | InvalidTypeException e) {
            assertEquals("invalid document name: expects alphanumeric, length <= 10", e.getMessage());
        }
    }

    @Test
    public void testInvalidFileType() {
        try {
            new Document(dummyDir, "Doc1", "Content", "exe");
            fail("Expected InvalidTypeException");
        } catch (InvalidFileException e) {
            fail("Unexpected InvalidFileException");
        } catch (InvalidTypeException e) {
            assertTrue(e.getMessage().contains("expects file type"));
        }
    }

    @Test
    public void fuzzTestDocumentCreation() {
        Random rand = new Random();
        for(int i = 0; i < 100000; i++) {
            String docName = getRandStr(rand, rand.nextInt(15));
            String content = getRandStr(rand, rand.nextInt(100));
            String type = getRandFileType(rand);
            try {
                Document doc = new Document(dummyDir, docName, content, type);
                assertEquals(docName, doc.getDocName());
                assertEquals(FileType.getFileType(type), doc.getType());
                assertEquals(40 + ((long)content.length() << 1), doc.getSize());
                assertEquals(dummyDir, doc.getPa());
            } catch (InvalidFileException | InvalidTypeException e) {
                // Expected for invalid inputs
                if (docName.length() >10 || !docName.matches("[A-Za-z0-9]+")) {
                    assertTrue(e instanceof InvalidFileException);
                } else if (notValidFileType(type)) {
                    assertTrue(e instanceof InvalidTypeException);
                } else {
                    fail("Unexpected exception: "+e);
                }
            }
        }
    }
}