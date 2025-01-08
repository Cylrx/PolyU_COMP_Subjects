package hk.edu.polyu.comp.comp2021.cvfs.model.criterion;

import hk.edu.polyu.comp.comp2021.cvfs.exceptions.InvalidCriterionException;
import hk.edu.polyu.comp.comp2021.cvfs.exceptions.InvalidValueException;
import hk.edu.polyu.comp.comp2021.cvfs.model.Directory;
import hk.edu.polyu.comp.comp2021.cvfs.model.Document;

import java.io.Serializable;

import static hk.edu.polyu.comp.comp2021.cvfs.utils.TypeUtils.isNumber;
import static hk.edu.polyu.comp.comp2021.cvfs.utils.TypeUtils.toNumber;

/**
 * A criterion that compares the size of a document or directory with a target size.
 * The size can be compared with the following operators:
 * <ul>
 *  <li>&lt; (less than)</li>
 *  <li>&gt; (greater than)</li>
 *  <li>== (equals to)</li>
 *  <li>&lt;= (less than or equals to)</li>
 *  <li>&gt;= (greater than or equals to)</li>
 *  <li>!= (not equals to)</li>
 * </ul>
 */
public class SizeCriterion implements Criterion, Serializable {
    // [size] { <, >, <=, >=, ==, != } target
    private final long target;
    private final Comparator cmp;
    private final String crtName;

    private enum Comparator{
        L, // Less than
        G, // Greater than
        E, // Equals to
        LE, // Less than OR equals to
        GE, // Greater than OR equals to
        NE; // Not equals to

        private boolean eval (long size, long target) {
            switch (this) {
                case L: return size < target;
                case G: return size > target;
                case E: return size == target;
                case LE: return size <= target;
                case GE: return size >= target;
                default: return size != target; // NE
            }
        }

        private String getExpr(long target) {
            String template = "(size %s %d)";
            switch (this) {
                case L: return String.format(template, "<", target);
                case G: return String.format(template, ">", target);
                case E: return String.format(template, "==", target);
                case LE: return String.format(template, "<=", target);
                case GE: return String.format(template, ">=", target);
                default: return String.format(template, "!=", target); // NE
            }
        }
    }

    /**
     * @param crtName the name of the criterion (must be precisely two alphabets)
     * @param op the operator of the criterion (must be one of the following: <, >, <=, >=, !=, ==)
     * @param val the target size to compare with
     * @throws InvalidCriterionException when the operator is invalid
     * @throws InvalidValueException when the value is not a number
     */
    public SizeCriterion(String crtName, String op, String val) throws InvalidCriterionException, InvalidValueException {
        assertValidArgs(op, val);
        this.cmp = toComparator(op);
        this.target = toNumber(val);
        this.crtName = crtName;
    }

    private static void assertValidArgs(String op, String val) throws InvalidCriterionException {
        if (!isComparator(op)) throw Criterion.getInvalidCriterionException("opcode", "{<, >, <=, >=, !=, ==}", op);
        if (!isNumber(val)) throw Criterion.getInvalidCriterionException("operand", "integer", val);
    }

    @Override
    public String getCrtName() {return crtName;}
    @Override
    public String getExpr() {return cmp.getExpr(target);}
    @Override
    public boolean eval (Document doc) { return cmp.eval(doc.getSize(), target); }
    @Override
    public boolean eval (Directory dir) { return cmp.eval(dir.getSize(), target); }

    private static boolean isComparator (String cmp) {
        switch (cmp) {
            case "<": case ">": case "<=": case ">=": case "!=": case "==":
                return true;
        }
        return false;
    }

    private static Comparator toComparator (String cmp) {
        switch (cmp) {
            case "<": return Comparator.L;
            case ">": return Comparator.G;
            case "==": return Comparator.E;
            case "<=": return Comparator.LE;
            case ">=": return Comparator.GE;
            default: return Comparator.NE; // !=
        }
    }
}
