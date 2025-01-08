library(ggplot2) # Load the ggplot2 library

# Define the kmean function for k-means clustering
# Usage: 	data = data matrix (1 feature per row, 1 sample per column),
#			iteration = number of optimization steps
# 			k = number of clusteres,
# 			centroids = centroid matrix (1 feature per row, 1 centroid per column)
# 			tol = maximum tolerable absolute distance between centroids at each iteration (for early stopping)

kmean <- function(data, iteration = 100, k = 5, centroids, tol = 0) {
	data <- t(data) # Transpose data matrix such that there is 1 sample per column (matching the centroids matrix)
	loss <- numeric(0) # Initialize an empty vector to store loss values
	for (i in 1:iteration) {
		pre_centroids <- centroids # Stores the previous centroids before update
		dists <- sapply(1:k, function(j) colSums((data - centroids[, j])^2)) # Compute distance matrix between each centroid and all samples
		groups <- apply(dists, 1, which.min) # Assign each sample to their nearest cluster 
		clusters <- sapply(1:k, function(j) data[, groups == j]) # Organize samples of the same cluster into the same sublists
		centroids <- sapply(1:k, function(j) rowMeans(clusters[[j]])) # Recalculate the centroids
		this_loss <- unlist(sapply(1:k, function(j) dists[groups == j, j])) # Compute average distance
		loss <- c(loss, mean(this_loss)) # Store average distance of this iteration into the loss vector
		if(sum(abs(centroids-pre_centroids)) <= tol) break
		# if absolute distance between previous centroids and updated centroids is less than tolerable value (0 by default), early stop.
	}
	return(list(centroids = centroids, groups = groups, loss = loss)) # Return results as a list
}

# Graph plotting function for plotting the loss curve and the clusters
# Usage:	data_points = dataframe of the samples. Must include columns 'Annual_Income', 'Spending_Score', and 'groups'
# 			centroids = dataframe of centroids. Must include columns 'x', 'y', representing the (x,y) coordinates of the centroids
# 			loss = dataframe of the loss. Must include columns 'it' (iteration count from 1~N) and 'loss' (mean distance to clusters at each iteration).

plot_graph <- function(data_points, centroids, loss) {
	g_cluster <- ggplot() + # Create ggplot for scatterplot of the samples & centroids
		geom_point(data = data_points, mapping = aes(x = Annual_Income, y = Spending_Score, col = as.factor(groups))) +
		scale_color_manual(values = c('red', 'green', 'blue', 'purple', 'brown')) +
		geom_point(data = centroids, mapping = aes(x = x, y = y), shape = 17, size = 3) +
		labs(x = 'Annual Income', y = 'Spending Score', color = 'Clusters')
	g_loss <- ggplot() + # Create ggplot for the loss curve
		geom_line(data = loss, mapping = aes(x = it, y = loss)) +
		labs(x = 'Iterations', y = 'Mean Distance to Centroids', title = 'Loss Curve')
	print(g_cluster) # Display scatterplot
	print(g_loss) # Display loss curve
}

main <- function() {
	df <- read.csv('./Customers.csv') # Read from csv file
	data <- cbind(df$Annual_Income, df$Spending_Score) # Combine Annual_Income and Spending_Store into a separate data matrix
	centroids <- matrix(c(5, 5, 20, 90, 100, 10, 70, 50, 120, 100), nrow = 2)  # Initialize centroids as a matrix (1 centroid / column)
	res <- kmean(data, k = 5, centroids = centroids) # Get kmean clustering result
	data_points <- cbind(df, groups = res$groups) # Combine clustering result with the original dataframe
	centroids <- data.frame(x = res$centroids[1,], y = res$centroids[2,]) # Combine centroids into a dataframe
	loss <- data.frame(it = seq_along(res$loss), loss = res$loss) # Convert loss value into a dataframe
	plot_graph(data_points, centroids, loss) # Plot graph
}

main()

