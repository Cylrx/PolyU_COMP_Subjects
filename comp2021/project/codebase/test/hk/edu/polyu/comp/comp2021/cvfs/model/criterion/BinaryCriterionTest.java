package hk.edu.polyu.comp.comp2021.cvfs.model.criterion;

import hk.edu.polyu.comp.comp2021.cvfs.exceptions.InvalidCriterionException;
import hk.edu.polyu.comp.comp2021.cvfs.exceptions.InvalidFileException;
import hk.edu.polyu.comp.comp2021.cvfs.exceptions.InvalidTypeException;
import hk.edu.polyu.comp.comp2021.cvfs.exceptions.InvalidValueException;
import hk.edu.polyu.comp.comp2021.cvfs.model.Document;
import hk.edu.polyu.comp.comp2021.cvfs.model.Directory;
import org.junit.*;
import java.util.Random;

import static hk.edu.polyu.comp.comp2021.cvfs.model.TestUtils.*;
import static org.junit.Assert.*;

public class BinaryCriterionTest {

    // Static Tests
    @Test
    public void testValidBinaryCriterionAnd() throws InvalidCriterionException, InvalidValueException, InvalidFileException, InvalidTypeException {
        NameCriterion c1 = new NameCriterion("C1", "contains", "\"Test\"");
        SizeCriterion c2 = new SizeCriterion("C2", ">", "100");
        BinaryCriterion bc = new BinaryCriterion("BC", "&&", c1, c2);

        assertEquals("BC", bc.getCrtName());
        assertEquals("((name contains \"Test\") && (size > 100))", bc.getExpr());

        Document doc1 = new Document(null, "TestDoc", repeat("a", 31), "txt"); // size = 40 + 62 = 102
        Document doc2 = new Document(null, "TestDoc", repeat("a", 30), "txt"); // size = 100
        Document doc3 = new Document(null, "SampleDoc", repeat("a", 31), "java");
        Directory dir1 = new Directory("TestDir", null);

        assertTrue(bc.eval(doc1));
        assertFalse(bc.eval(doc2)); // size not > 100
        assertFalse(bc.eval(doc3)); // name does not contain "Test"
        assertFalse(bc.eval(dir1)); // both criteria fail
    }

    @Test
    public void testValidBinaryCriterionOr() throws InvalidCriterionException, InvalidValueException, InvalidTypeException, InvalidFileException {
        NameCriterion c1 = new NameCriterion("C1", "contains", "\"Data\"");
        TypeCriterion c2 = new TypeCriterion("C2", "equals", "\"txt\"");
        BinaryCriterion bc = new BinaryCriterion("BC", "||", c1, c2);

        assertEquals("BC", bc.getCrtName());
        assertEquals("((name contains \"Data\") || (type equals txt))", bc.getExpr());

        Document doc1 = new Document(null, "DataReport", "Content", "java");
        Document doc2 = new Document(null, "Summary", "Content", "txt");
        Document doc3 = new Document(null, "Sample", "Content", "java");
        Directory dir1 = new Directory("DataDir", null);
        Directory dir2 = new Directory("OtherDir", null);

        assertTrue(bc.eval(doc1)); // name contains "Data"
        assertTrue(bc.eval(doc2)); // type equals "txt"
        assertFalse(bc.eval(doc3)); // neither
        assertTrue(bc.eval(dir1)); // name contains "Data"
        assertFalse(bc.eval(dir2)); // neither
    }

    @Test
    public void testInvalidBinaryOperator() {
        try {
            NameCriterion c1 = new NameCriterion("C1", "contains", "\"Test\"");
            SizeCriterion c2 = new SizeCriterion("C2", ">", "50");
            new BinaryCriterion("BC", "!", c1, c2);
            fail("Expected InvalidCriterionException due to invalid binary operator");
        } catch (InvalidCriterionException e){
            // skip
        } catch (InvalidValueException e){
            fail("Unexpected InvalidValueException");
        }
    }

    // Fuzz Tests
    @Test
    public void fuzzTestBinaryCriterionEval() throws InvalidCriterionException, InvalidValueException, InvalidFileException, InvalidTypeException {
        Random rand = new Random();
        String[] comparators = {"<", ">", "<=", ">=", "==", "!="};
        int T = 300, docT = 0, dirT = 0;

        for (int t = 0; t < T; t++) {
            String chosenNamePattern = "\"" + getRandStr(rand, 1) + "\"";
            String chosenComparator = comparators[rand.nextInt(comparators.length)];
            int sizeTarget = rand.nextInt(500) + 1;

            NameCriterion c1 = new NameCriterion("FuzzC1", "contains", chosenNamePattern);
            SizeCriterion c2 = new SizeCriterion("FuzzC2", chosenComparator, String.valueOf(sizeTarget));
            String operator = rand.nextBoolean() ? "&&" : "||";
            BinaryCriterion bc = new BinaryCriterion("FuzzBC", operator, c1, c2);

            String namePattern = chosenNamePattern.substring(1, chosenNamePattern.length() - 1); // Remove quotes

            for (int i = 0; i < T; i++) {

                // rand name, rand size, rand type
                String name = getRandStr(rand, rand.nextInt(10) + 1);
                int contentLength = rand.nextInt(500); // Determines size
                long size = 40 + ((long) contentLength * 2);
                String type = getRandFileType(rand);

                // rand doc & dir, using the rand name, size, and type
                Document doc = new Document(null, name, getRandStr(rand, contentLength), type);
                Directory dir = new Directory(name, null);

                // test the criteria on the rand doc & dir
                boolean c1EvalDoc = name.contains(namePattern);
                boolean c2EvalDoc = compare(size, chosenComparator, sizeTarget);
                boolean expectedDoc = operator.equals("&&") ? (c1EvalDoc && c2EvalDoc) : (c1EvalDoc || c2EvalDoc);
                if (expectedDoc) docT++;

                boolean c1EvalDir = dir.getDirName().contains(namePattern);
                boolean c2EvalDir = compare(dir.getSize(), chosenComparator, sizeTarget);
                boolean expectedDir = operator.equals("&&") ? (c1EvalDir && c2EvalDir) : (c1EvalDir || c2EvalDir);
                if (expectedDir) dirT++;

                assertEquals(expectedDoc, bc.eval(doc));
                assertEquals(expectedDir, bc.eval(dir));
            }
        }

        System.out.printf("%d matched directories out of %d trials\n", dirT, T * T);
        System.out.printf("%d matched documents out of %d trials\n", docT, T * T);
    }

}