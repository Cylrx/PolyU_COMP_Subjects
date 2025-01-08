package service;

import utils.validator.NameValidator;
import utils.validator.*;
import view.View;
import java.util.HashSet;

public enum MealAttr implements Attr{
    DISH_NAME("Dish_name", "Dish name", new NameValidator()),
    BIN(
            BanquetAttr.BIN.getAttrName(),
            BanquetAttr.BIN.getDescription(),
            BanquetAttr.BIN.getValidator()
    ),
    TYPE("Type", "Type", new NameValidator()),
    CUISINE("Cuisine", "Cuisine", new NameValidator()),
    PRICE("Price", "Price", new IntValidator());

    private final String attrName;
    public final String description; // user friendly description
    private final Validator vd;
    private final static Tables table = Tables.MEAL;

    MealAttr(String attrName, String description, Validator vd) {
        this.attrName = attrName;
        this.description = description;
        this.vd = vd;
    }

    public static String[][] getValidMealSet(String banquetID) {
        HashSet<String> usedNames = new HashSet<>();
        String[][] mealSet = new String[4][];
        for (int i = 0; i < 4; i++) {
            String mealName;
            while (true) {
                mealName = MealAttr.DISH_NAME.inputNewVal();
                if (!usedNames.contains(mealName)) {
                    usedNames.add(mealName);
                    break;
                }
                View.displayError("Meal name already exists");
            }
            String mealType = MealAttr.TYPE.inputNewVal();
            String mealCuisine = MealAttr.CUISINE.inputNewVal();
            String mealPrice = MealAttr.PRICE.inputNewVal();
            mealSet[i] = new String[]{mealName, banquetID, mealType, mealCuisine, mealPrice};
        }
        return mealSet;
    }

    @Override public Tables getTable() { return table; }
    @Override public String getAttrName() { return attrName; }
    @Override public String getDescription() { return description; }
    @Override public Validator getValidator() { return vd; }
    @Override public boolean isUpdatable() { return this != BIN && this != DISH_NAME; }
}
