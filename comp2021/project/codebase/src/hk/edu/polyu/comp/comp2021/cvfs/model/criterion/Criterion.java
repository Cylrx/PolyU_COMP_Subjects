package hk.edu.polyu.comp.comp2021.cvfs.model.criterion;

import hk.edu.polyu.comp.comp2021.cvfs.exceptions.InvalidCriterionException;
import hk.edu.polyu.comp.comp2021.cvfs.model.Directory;
import hk.edu.polyu.comp.comp2021.cvfs.model.Document;


/**
 * Interface representing different criteria for evaluation.
 * <p>Each criterion must implement the following methods:</p>
 * <ul>
 *   <li><code>boolean eval(Document doc)</code>: Evaluates whether a document meets the criterion.</li>
 *   <li><code>boolean eval(Directory dir)</code>: Evaluates whether a directory meets the criterion.</li>
 *   <li><code>String getCrtName()</code>: Returns the name of the criterion.</li>
 *   <li><code>String getExpr()</code>: Returns the string expression of the criterion.</li>
 * </ul>
 */
public interface Criterion {
    /**
     * @param doc the <code>Document</code> object to be evaluated
     * @return <code>true</code> if the document meets the criterion, <code>false</code> otherwise
     */
    boolean eval (Document doc);

    /**
     * @param dir the <code>Directory</code> object to be evaluated
     * @return <code>true</code> if the directory meets the criterion, <code>false</code> otherwise
     */
    boolean eval (Directory dir);

    /**
     * @return the unique 2 letter identifier name of the criterion
     */
    String getCrtName();

    /**
     * @return the string expression of the criterion
     */
    String getExpr();

    /**
     * @param component the component of the criterion command that is invalid
     * @param expected the expected value of the component
     * @param actual the actual received (invalid) value of the component
     * @return an <code>InvalidCriterionException</code> with the formatted error message
     */
    static InvalidCriterionException getInvalidCriterionException(
            String component,
            String expected,
            String actual
    ) {
        String errorMessage = String.format(
                "invalid %s: expects %s, got %s",
                component, expected, actual
        );
        return new InvalidCriterionException(errorMessage);
    }
}
