package hk.edu.polyu.comp.comp2021.cvfs.model.criterion;

import hk.edu.polyu.comp.comp2021.cvfs.model.Directory;
import hk.edu.polyu.comp.comp2021.cvfs.model.Document;

import java.io.Serializable;

/**
 * A criterion that negates another criterion.
 * For example, if the original criterion is "A && B", the negated criterion will be "!(A && B)".
 */
public class NegCriterion implements Criterion, Serializable {
    private final Criterion c;
    private final String crtName;

    /**
     * @param crtName the name of the criterion (must be precisely two alphabets)
     * @param c the criterion to be negated
     */
    public NegCriterion(String crtName, Criterion c) {
        this.crtName = crtName;
        this.c = c;
    }

    @Override
    public String getCrtName() {return crtName;}
    @Override
    public String getExpr() { return "!" + c.getExpr(); }
    @Override
    public boolean eval(Document doc) { return !c.eval(doc); }
    @Override
    public boolean eval(Directory dir) { return !c.eval(dir); }
}
