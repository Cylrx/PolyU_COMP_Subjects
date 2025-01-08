package hk.edu.polyu.comp.comp2021.cvfs.model.criterion;

import hk.edu.polyu.comp.comp2021.cvfs.exceptions.InvalidCriterionException;
import hk.edu.polyu.comp.comp2021.cvfs.exceptions.InvalidTypeException;
import hk.edu.polyu.comp.comp2021.cvfs.model.Directory;
import hk.edu.polyu.comp.comp2021.cvfs.model.Document;
import hk.edu.polyu.comp.comp2021.cvfs.model.FileType;

import java.io.Serializable;

/**
 * A criterion that checks if the filetype of a document is equal to a specified one.
 * The eval() method only works with Document objects. It will always return false for Directory objects.
 * The type must be one of the following: {txt, java, html, css}
 */
public class TypeCriterion implements Criterion, Serializable {
    // [type] equals "type"

    private final String crtName;
    private final FileType type;

    /**
     * @param crtName the name of the criterion (must be precisely two alphabets)
     * @param op the operator of the criterion (must be "equals")
     * @param val the type we wish to check for in the file types (must be one of the following: {txt, java, html, css})
     * @throws InvalidCriterionException when the operator or value is invalid
     * @throws InvalidTypeException when the type is invalid
     */
    public TypeCriterion(String crtName, String op, String val) throws InvalidCriterionException, InvalidTypeException {
        assertValidArgs(op, val);
        String type = val.substring(1, val.length() - 1);
        this.type = FileType.getFileType(type);
        this.crtName = crtName;
    }

    private static void assertValidArgs(String op, String val) throws InvalidCriterionException {
        if (!op.equals("equals")) {
            throw Criterion.getInvalidCriterionException("opcode", "\"equals\"", op);
        }

        if (val.length() < 2 || !val.startsWith("\"") || !val.endsWith("\"")) {
            throw Criterion.getInvalidCriterionException("operand", "a double quoted string", val);
        }
    }

    @Override
    public String getCrtName() {
        return crtName;
    }

    @Override
    public String getExpr() {
        return String.format("(type equals %s)", type.toStrType());
    }

    @Override
    public boolean eval(Document doc) {
        return type == doc.getType();
    }

    @Override
    public boolean eval(Directory dir) {
        return false;
    }
}
