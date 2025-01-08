package hk.edu.polyu.comp.comp2021.cvfs.model.criterion;

import hk.edu.polyu.comp.comp2021.cvfs.exceptions.InvalidCriterionException;
import hk.edu.polyu.comp.comp2021.cvfs.exceptions.InvalidFileException;
import hk.edu.polyu.comp.comp2021.cvfs.exceptions.InvalidTypeException;
import hk.edu.polyu.comp.comp2021.cvfs.model.Document;
import hk.edu.polyu.comp.comp2021.cvfs.model.FileType;
import hk.edu.polyu.comp.comp2021.cvfs.model.Directory;
import org.junit.*;
import java.util.Random;

import static hk.edu.polyu.comp.comp2021.cvfs.model.TestUtils.getRandFileType;
import static hk.edu.polyu.comp.comp2021.cvfs.model.TestUtils.getRandStr;
import static org.junit.Assert.*;

public class TypeCriterionTest {

    // Static Tests
    @Test
    public void testValidTypeCriterion() throws InvalidCriterionException, InvalidTypeException, InvalidFileException {
        TypeCriterion tc = new TypeCriterion("TypeCrt", "equals", "\"txt\"");
        assertEquals("TypeCrt", tc.getCrtName());
        assertEquals("(type equals txt)", tc.getExpr());

        Document doc1 = new Document(null, "Doc1", "Content", "txt");
        Document doc2 = new Document(null, "Doc2", "Content", "java");
        Directory dir1 = new Directory("Dir1", null);

        assertTrue(tc.eval(doc1));
        assertFalse(tc.eval(doc2));
        assertFalse(tc.eval(dir1));
    }

    @Test
    public void testInvalidOpcode() {
        try {
            new TypeCriterion("TypeCrt", "contains", "\"txt\"");
            fail("Expected InvalidCriterionException due to invalid opcode");
        } catch (InvalidCriterionException | InvalidTypeException e){
            assertEquals("invalid opcode: expects \"equals\", got contains", e.getMessage());
        }
    }

    @Test
    public void testInvalidOperand() {
        try {
            new TypeCriterion("TypeCrt", "equals", "txt");
            fail("Expected InvalidCriterionException due to missing quotes");
        } catch (InvalidCriterionException | InvalidTypeException e){
            assertEquals("invalid operand: expects a double quoted string, got txt", e.getMessage());
        }
    }

    @Test
    public void testInvalidFileType() {
        try {
            new TypeCriterion("TypeCrt", "equals", "\"exe\"");
            fail("Expected InvalidTypeException due to invalid file type");
        } catch (InvalidCriterionException e){
            fail("Unexpected InvalidCriterionException");
        } catch (InvalidTypeException e){
            assertTrue(e.getMessage().contains("expects file type"));
        }
    }

    // Fuzz Tests
    @Test
    public void fuzzTestFileType() {
        Random rand = new Random();
        int T = 5000;
        for (int t = 0; t < T; t++) {
            boolean isValid = rand.nextBoolean();
            boolean isLQuoted = rand.nextBoolean();
            boolean isRQuoted = rand.nextBoolean();
            String name = (isValid) ? getRandFileType(rand) : getRandStr(rand, rand.nextInt(5));
            String finalName = name;
            finalName = (isLQuoted) ? "\"" + name : finalName;
            finalName = (isRQuoted) ? finalName + "\"" : finalName;

            try {
                new TypeCriterion("tc", "equals", finalName);
                if (!isValid || !isLQuoted || !isRQuoted) fail("Expected InvalidTypeException: " + name + "is not a valid type");
            } catch (InvalidCriterionException e) {
                if (isLQuoted && isRQuoted) fail("Unexpected InvalidCriterionException");
            } catch (InvalidTypeException e) {
                if (isValid) fail("Unexpected InvalidTypeException: " + name + " is a valid type");
                else assertTrue(e.getMessage().contains("expects file type"));
            }
        }
    }

    @Test
    public void fuzzTestTypeCriterionEval() throws InvalidCriterionException, InvalidTypeException, InvalidFileException {
        Random rand = new Random();
        FileType[] validTypes = FileType.values();
        int T = 1000, tCnt = 0;
        for (int t = 0; t < T; t++) {
            FileType chosenType = validTypes[rand.nextInt(validTypes.length)];
            TypeCriterion tc = new TypeCriterion("FuzzTypeCrt", "equals", "\"" + chosenType.toStrType() + "\"");

            for (int i = 0; i < 10; i++) {
                FileType docType = validTypes[rand.nextInt(validTypes.length)];
                Document doc = new Document(null, "Doc" + i, "Content", docType.toStrType());
                Directory dir = new Directory("Dir" + i, null);

                boolean expected = (docType == chosenType);
                if (expected) tCnt++;

                assertEquals(expected, tc.eval(doc));
                assertFalse(tc.eval(dir));
            }
        }
        System.out.printf("%d document matches out of %d trials", tCnt, T * 10);
    }
}