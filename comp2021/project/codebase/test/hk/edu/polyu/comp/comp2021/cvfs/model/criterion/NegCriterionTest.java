package hk.edu.polyu.comp.comp2021.cvfs.model.criterion;

import hk.edu.polyu.comp.comp2021.cvfs.exceptions.*;
import hk.edu.polyu.comp.comp2021.cvfs.model.Document;
import hk.edu.polyu.comp.comp2021.cvfs.model.Directory;
import org.junit.*;
import java.util.Random;

import static hk.edu.polyu.comp.comp2021.cvfs.model.TestUtils.*;
import static org.junit.Assert.*;

public class NegCriterionTest {

    // Static Tests
    @Test
    public void testNegCriterion() throws InvalidCriterionException, InvalidFileException, InvalidTypeException {
        NameCriterion nc = new NameCriterion("InnerCrt", "contains", "\"Test\"");
        NegCriterion negCrt = new NegCriterion("NegCrt", nc);

        assertEquals("NegCrt", negCrt.getCrtName());
        assertEquals("!(name contains \"Test\")", negCrt.getExpr());

        Document doc1 = new Document(null, "TestDoc", "Content", "txt");
        Document doc2 = new Document(null, "LolDoc", "Content", "java");
        Directory dir1 = new Directory("TestDir", null);
        Directory dir2 = new Directory("BadDir", null);

        assertFalse(negCrt.eval(doc1));
        assertTrue(negCrt.eval(doc2));
        assertFalse(negCrt.eval(dir1));
        assertTrue(negCrt.eval(dir2));
    }

    // Fuzz Tests
    @Test
    public void fuzzTestNegCriterionEval() throws InvalidCriterionException, InvalidFileException, InvalidTypeException {
        Random rand = new Random();
        int T = 3000, tCnt = 0;
        for (int t = 0; t < T; t++) {
            String pattern = getRandStr(rand, rand.nextInt(3) + 1);
            NameCriterion nc = new NameCriterion("FuzzInnerCrt", "contains", "\"" + pattern + "\"");
            NegCriterion negCrt = new NegCriterion("FuzzNegCrt", nc);

            for (int i = 0; i < 100; i++) {
                String name = getRandStr(rand, rand.nextInt(10) + 1);
                boolean expected = !name.contains(pattern);
                if (expected) tCnt++;

                Document doc = new Document(null, name, "Content", "txt");
                Directory dir = new Directory(name, null);

                assertEquals(expected, negCrt.eval(doc));
                assertEquals(expected, negCrt.eval(dir));
            }
        }
        System.out.printf("%d matches out of %d trials\n", tCnt, T * T);
    }
}