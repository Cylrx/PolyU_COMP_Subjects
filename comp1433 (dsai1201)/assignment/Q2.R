library(ggplot2) # import ggplot2
set.seed(1234) # Set seed to 1234 as requested by the 2(a)
size <- 10000 # Sample Size
sd <- 0.05 # Standard Deviation
mu <- 0.02 # Mean

lnorm_samples <- exp(rnorm(size, mu, sd)) # Generate log-normal distribution samples by taking the exponential of normally distributed samples
lnorm_range <- seq(min(lnorm_samples), max(lnorm_samples), length.out = size) # Generate a sequence of evenly spaced numbers within the range of the log-normal samples
lnorm_density <- dlnorm(lnorm_range, mu, sd) # Map sequence generated above to density

# Question 2(a) - Display a histogram of the log-normal samples
question_2a <- function() {
	graph <- ggplot() +
		geom_histogram(mapping = aes(x = lnorm_samples), bins = 30) +
		labs(title = "Question 2(a) Graph", x = "x", y = "Count")
	print(graph)
}

# Question 2(b) - OVerlay the log-normal sample histogram with the theoretical density curve
question_2b <- function() {
	width <- (max(lnorm_samples) - min(lnorm_samples)) / 30 # Calculate the histogram bin width
	breaks <- seq(min(lnorm_samples), max(lnorm_samples), width) # Define breaks for the histogram (separation points)
	scale <- max(hist(lnorm_samples, breaks = breaks, plot = FALSE)$counts) / max(lnorm_density) # Calculate the scaling factor for the density curve to match histograwm
	graph <- ggplot() + # Plot graph
    	geom_histogram(mapping = aes(x = lnorm_samples), bins = 30) +
    	geom_line(mapping = aes(x = lnorm_range, y = lnorm_density * scale)) + # Scaling the density plot
    	scale_y_continuous(name = "Count", sec.axis = sec_axis(~. / scale, name = 'Density')) + # Creating a secondary y-axis for "density" in addition to "count"
    	labs(title = "Question 2(b) Graph", x = "x")
	print(graph)
}

# Question 2(c) - Print variance and mean of log-normal samples for different sample sizes
question_2c <- function() {
	cat("Question 2(c) Results: \n")
	sizes <- c(10, 100, 1000, 10000, 100000) # Different sample sizes to evaluate as requested by the 2(c)
	for (n in sizes) { # iterate over the sample sizes
		tmp_samples <- exp(rnorm(n, mu, sd)) # Generate log-normal samples for each sample size
		cat("Sample Size:", n, "- Var:", var(tmp_samples), "| Mean:", mean(tmp_samples), "\n") # Print sample variance and mean
	}
}

# Question 2(d)
question_2d <- function() {
	cat("\nQuestion 2(d) Results: \n")
	print(length(lnorm_samples[lnorm_samples >= 1.05]) / length(lnorm_samples))
	# Calculate and print the proportion of simulated samples greater than daily return of 1.05
}

question_2e <- function() {
	cat("\nQuestion 2(e) Results: \n")
	print(plnorm(1.05, mu, sd, lower.tail = FALSE))
	# Calculate and print probability of random variable of the given log-normal distribution greater than 1.05
}

# Execute all functions sequentially to display results
question_2a()
question_2b()
question_2c()
question_2d()
question_2e()

