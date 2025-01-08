package hk.edu.polyu.comp.comp2021.cvfs.model.criterion;

import hk.edu.polyu.comp.comp2021.cvfs.exceptions.InvalidCriterionException;
import hk.edu.polyu.comp.comp2021.cvfs.model.Directory;
import hk.edu.polyu.comp.comp2021.cvfs.model.Document;

import java.io.Serializable;

/**
 * A binary criterion that combines two criteria with a logical operator.
 * The logic operator can be either <code>&&</code> or <code>||</code>.
 */
public class BinaryCriterion implements Criterion, Serializable {
    private final Criterion c1, c2;
    private final LogicOperator op;
    private final String crtName;

    private enum LogicOperator {
        AND, OR;

        private boolean eval(boolean b1, boolean b2) {
            if (this == AND) return b1 && b2;
            return b1 || b2;
        }

        private String getExpr(String c1Str, String c2Str) {
            if (this == AND) return String.format("(%s && %s)", c1Str, c2Str);
            return String.format("(%s || %s)", c1Str, c2Str);
        }
    }

    /**
     * @param crtName the name of the criterion (must be precisely two alphabets)
     * @param op the operator of the binary criterion (must be || or &&)
     * @param c1 the first Criterion object
     * @param c2 the second Criterion object
     * @throws InvalidCriterionException when the operator is invalid
     */
    public BinaryCriterion(String crtName, String op, Criterion c1, Criterion c2) throws InvalidCriterionException{
        if (!isValidOp(op))
            throw Criterion.getInvalidCriterionException("operator", "&& or ||", op);
        this.crtName = crtName;
        this.op = toOp(op);
        this.c1 = c1;
        this.c2 = c2;
    }

    private static boolean isValidOp (String op) {
        return op.equals("&&") || op.equals("||");
    }
    private static LogicOperator toOp (String op) {
        if (op.equals("&&")) return LogicOperator.AND;
        return LogicOperator.OR;
    }

    @Override
    public String getCrtName() {return crtName;}
    @Override
    public String getExpr() {return op.getExpr(c1.getExpr(), c2.getExpr()); }
    @Override
    public boolean eval (Document doc) { return op.eval(c1.eval(doc), c2.eval(doc)); }
    @Override
    public boolean eval (Directory dir) { return op.eval(c1.eval(dir), c2.eval(dir)); }
}
