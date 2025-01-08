package hk.edu.polyu.comp.comp2021.cvfs.model;

import hk.edu.polyu.comp.comp2021.cvfs.exceptions.InvalidTypeException;
import org.junit.*;
        import static org.junit.Assert.*;

public class FileTypeTest {

    // Static Tests
    @Test
    public void testGetFileTypeValid() throws InvalidTypeException {
        assertEquals(FileType.TXT, FileType.getFileType("txt"));
        assertEquals(FileType.JAVA, FileType.getFileType("java"));
        assertEquals(FileType.HTML, FileType.getFileType("html"));
        assertEquals(FileType.CSS, FileType.getFileType("css"));
    }

    @Test
    public void testGetFileTypeInvalid() {
        try {
            FileType.getFileType("exe");
            fail("Expected InvalidTypeException");
        } catch (InvalidTypeException e) {
            assertTrue(e.getMessage().contains("expects file type"));
        }
    }

    @Test
    public void testToStrType() {
        assertEquals("txt", FileType.TXT.toStrType());
        assertEquals("java", FileType.JAVA.toStrType());
        assertEquals("html", FileType.HTML.toStrType());
        assertEquals("css", FileType.CSS.toStrType());
    }

    // Fuzz Tests
    @Test
    public void fuzzTestGetFileType() {
        String[] valid = {"txt","java","html","css"};
        String[] invalid = {"exe","pdf","", " ", "123", "htmll", "csss", "TXT","JAVA","HTML","CSS"," Txt "," java "};
        for(String type : valid){
            try {
                FileType ft = FileType.getFileType(type);
                assertNotNull(ft);
            } catch (InvalidTypeException e){
                fail("Unexpected InvalidTypeException for valid type: "+type);
            }
        }
        for(String type : invalid){
            try {
                FileType.getFileType(type);
                fail("expected InvalidTypeException for type: "+type);
            } catch (InvalidTypeException e){
                assertTrue(e.getMessage().contains("expects file type"));
            }
        }
    }
}