package hk.edu.polyu.comp.comp2021.cvfs.model.criterion;

import hk.edu.polyu.comp.comp2021.cvfs.model.Document;
import hk.edu.polyu.comp.comp2021.cvfs.model.Directory;

import java.io.Serializable;

/**
 * Criterion for filtering by document
 * Returns true if the object is a document
 */
public class IsDocCriterion implements Criterion, Serializable {
    @Override
    public boolean eval (Document doc) { return true; }

    @Override
    public boolean eval (Directory dir) { return false;}

    @Override
    public String getCrtName() {
        return "IsDocument";
    }

    @Override
    public String getExpr() {
        return getCrtName();
    }
}
