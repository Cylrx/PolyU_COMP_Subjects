package hk.edu.polyu.comp.comp2021.cvfs.model.criterion;

import hk.edu.polyu.comp.comp2021.cvfs.exceptions.InvalidCriterionException;
import hk.edu.polyu.comp.comp2021.cvfs.model.Directory;
import hk.edu.polyu.comp.comp2021.cvfs.model.Document;

import java.io.Serializable;

/**
 * Criterion for filtering by name
 */
public class NameCriterion implements Criterion, Serializable {
    // [name] contains "..."

    private final String pattern;
    private final String crtName;

    /**
     * @param crtName the name of the criterion (must be precisely two alphabets)
     * @param op the operator of the criterion (must be "contains")
     * @param val the substring we wish to check for in the file names
     * @throws InvalidCriterionException when the operator or value is invalid
     */
    public NameCriterion(String crtName, String op, String val) throws InvalidCriterionException{
        assertValidArgs(op, val);
        this.pattern = val.substring(1, val.length() - 1);
        this.crtName = crtName;
    }

    private static void assertValidArgs(String op, String val) throws InvalidCriterionException {
        if (!op.equals("contains")) {
            throw Criterion.getInvalidCriterionException("opcode", "\"contains\"", op);
        }
        if (val.length() < 2 || !val.startsWith("\"") || !val.endsWith("\"") ) {
            throw Criterion.getInvalidCriterionException("operand", "a double quoted string", val);
        }
    }

    @Override
    public String getCrtName() {return crtName; }
    @Override
    public String getExpr() {return String.format("(name contains \"%s\")", pattern); }
    @Override
    public boolean eval (Document doc) { return doc.getDocName().contains(pattern); }
    @Override
    public boolean eval (Directory dir) { return dir.getDirName().contains(pattern); }
}
