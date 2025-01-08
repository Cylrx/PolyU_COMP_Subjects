package hk.edu.polyu.comp.comp2021.cvfs.model.criterion;

import hk.edu.polyu.comp.comp2021.cvfs.exceptions.InvalidCriterionException;
import hk.edu.polyu.comp.comp2021.cvfs.exceptions.InvalidFileException;
import hk.edu.polyu.comp.comp2021.cvfs.exceptions.InvalidTypeException;
import hk.edu.polyu.comp.comp2021.cvfs.model.Document;
import hk.edu.polyu.comp.comp2021.cvfs.model.Directory;
import org.junit.*;
import java.util.Random;

import static org.junit.Assert.*;
import static hk.edu.polyu.comp.comp2021.cvfs.model.TestUtils.*;

public class NameCriterionTest {

    // Static Tests
    @Test
    public void testValidNameCriterion() throws InvalidCriterionException, InvalidFileException, InvalidTypeException {
        NameCriterion nc = new NameCriterion("NameCrt", "contains", "\"Test\"");
        assertEquals("NameCrt", nc.getCrtName());
        assertEquals("(name contains \"Test\")", nc.getExpr());

        Document doc1 = new Document(null, "TestDoc1", "Content", "txt");
        Document doc2 = new Document(null, "LolDoc2", "Content", "java");
        Directory dir1 = new Directory("TestDir1", null);
        Directory dir2 = new Directory("LolDir2", null);

        assertTrue(nc.eval(doc1));
        assertFalse(nc.eval(doc2));
        assertTrue(nc.eval(dir1));
        assertFalse(nc.eval(dir2));
    }

    @Test
    public void testInvalidOpcode() {
        try {
            new NameCriterion("NameCrt", "equals", "\"Test\"");
            fail("Expected InvalidCriterionException due to invalid opcode");
        } catch (InvalidCriterionException e){
            assertEquals("invalid opcode: expects \"contains\", got equals", e.getMessage());
        }
    }

    @Test
    public void testInvalidOperand() {
        try {
            new NameCriterion("NameCrt", "contains", "Test");
            fail("Expected InvalidCriterionException due to missing quotes");
        } catch (InvalidCriterionException e){
            assertEquals("invalid operand: expects a double quoted string, got Test", e.getMessage());
        }
    }

    // Fuzz Tests
    @Test
    public void fuzzTestPattern() {
        Random rand = new Random();
        int T = 2000;
        for (int t = 0; t < T; t++) {
            String pattern = getRandStr(rand, rand.nextInt(5));
            boolean isLQuoted = rand.nextBoolean();
            boolean isRQuoted = rand.nextBoolean();
            pattern = (isLQuoted) ? "\"" + pattern : pattern;
            pattern = (isRQuoted) ? pattern + "\"" : pattern;

            try {
                NameCriterion nc = new NameCriterion("nc", "contains", pattern);
                if (!isLQuoted || !isRQuoted) fail("Expected InvalidCriterionException: double quote missing");
            } catch (InvalidCriterionException e) {
                if (isLQuoted && isRQuoted) fail ("Unexpected InvalidCriterionException: double quoted");
            }

        }
    }

    @Test
    public void fuzzTestNameCriterionEval() throws InvalidCriterionException, InvalidFileException, InvalidTypeException {
        Random rand = new Random();
        int T = 2000, tCnt = 0;
        for (int t = 0; t < T; t++) {
            String pattern = getRandStr(rand, rand.nextInt(3) + 1);
            NameCriterion nc = new NameCriterion("FuzzNameCrt", "contains", "\"" + pattern + "\"");

            for (int i = 0; i < T; i++) {
                String name = getRandStr(rand, rand.nextInt(10) + 1);
                boolean expected = name.contains(pattern);
                if (expected) tCnt++;

                Document doc = new Document(null, name, "Some Content", "txt");
                Directory dir = new Directory(name, null);

                assertEquals(expected, nc.eval(doc));
                assertEquals(expected, nc.eval(dir));
            }
        }
        System.out.printf("%d matches out of %d trials\n", tCnt, T * T);
    }
}