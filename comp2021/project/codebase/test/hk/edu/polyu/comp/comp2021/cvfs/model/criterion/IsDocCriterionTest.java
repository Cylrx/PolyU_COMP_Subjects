package hk.edu.polyu.comp.comp2021.cvfs.model.criterion;

import hk.edu.polyu.comp.comp2021.cvfs.exceptions.InvalidFileException;
import hk.edu.polyu.comp.comp2021.cvfs.exceptions.InvalidTypeException;
import hk.edu.polyu.comp.comp2021.cvfs.model.Directory;
import hk.edu.polyu.comp.comp2021.cvfs.model.Document;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

public class IsDocCriterionTest {
    private final static IsDocCriterion isDocCriterion = new IsDocCriterion();
    private Document testDoc;
    private Directory testDir;

    @Before
    public void setUp() throws InvalidFileException, InvalidTypeException {
        testDir = new Directory("dir1", null);
        testDoc = new Document(testDir, "doc1", "content", "txt");
    }

    @Test
    public void testEval() throws InvalidFileException, InvalidTypeException {
        assertTrue(isDocCriterion.eval(testDoc));
        assertFalse(isDocCriterion.eval(testDir));
    }

    @Test
    public void testGetCrtName() {
        assertEquals("IsDocument", isDocCriterion.getCrtName());
    }

    @Test
    public void testGetExpr() {
        assertEquals("IsDocument", isDocCriterion.getExpr());
    }
}