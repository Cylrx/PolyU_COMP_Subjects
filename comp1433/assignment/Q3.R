library(ggplot2) # Load ggplot

data <- read.csv('./house_prices_dataset.csv')
response <- 'House_Price'
predictors <- setdiff(names(data),  response)




# Question 3(a)
# Loop through each predictor variable (pred in {predictors})
#   - title (String type): Plot's title
#   - equation (String type): the fitted linear equation
#   - model: The linear model
#   - coef: parameters of the fitted model (intercept & slope)

for(pred in predictors){
	title <- paste(response, 'vs', pred) 
	model <- lm(formula(paste(response, '~', pred)), data) 
	coef <- coefficients(model) 
	equation <- paste(response, '=', round(coef[2], 1), '*', pred, '+', round(coef[1], 1)) 

	graph <- ggplot() +
		geom_point(aes(x = data[[pred]], y = data[[response]]), col="cornflowerblue") + # Plot Scatterplot of the datapoints
		geom_abline(slope = coef[2], intercept = coef[1], col = "darkblue", linewidth=0.9) + # Overlay the regression line
		annotate("text", x = median(data[[pred]]), y = median(data[[response]]), label = equation, vjust=-10, size=5) + # Display the equation on top of the regression line
		labs(title = title, x = pred, y = response) # Label the graph
	print(graph)
}




# Question 3(b)
# model: Multivariate linear model using the two predictors specified in 3(b)
# input: create input data for prediction as specified in 3(b)

model <- lm(House_Price ~ House_Area + Distance_to_Center, data = data) 
input <- data.frame(House_Area = 250, Distance_to_Center = 5) 
cat('Question 3(b): \n', response, '=', predict(model, newdata = input), '\n') 




# Question 3(c)
# max_r: tracks the highest R-squared value so far
# val_r: an array (or vector) of R-squared values for each pair of predictors
# index: The number of the model
# this_r: Stores the R-squared value of the current pair of predictors
# best_model: Stores the best fit model so far

# Control Flow
# Iterates over each predictor in the list (pred1 in {predictors})
#   - removes pred1 from predictors since it is already used
#   - iterates over the remaining (unused predictors) as pred2 (pred2 in {predictors} - pred1)
#   - fit linear model of response w.r.t pred1 & pred2
#   - If new results (R-squared) better than previous ones, update.

max_r <- -Inf 
vals_r <- numeric(0) 
index <- 0 
for(pred1 in predictors){
	predictors <- setdiff(predictors, pred1) 
	for(pred2 in predictors){ 
		model <- lm(formula(paste(response, '~', pred1, '+', pred2)), data = data) 
		this_r <- summary(model)$r.squared 
		vals_r <- c(vals_r, this_r) 
		index <- index + 1 
		if (this_r > max_r){ 
			max_r <- this_r 
			best_model <- index 
		}
	}
}
# Print results
cat('\nQuestion 3(c): \n')
cat('The R-squared values for all the models are', vals_r, '. So the best model is Model', best_model)

