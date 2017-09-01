package com.sandc;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

/**
 * Created by kenneth.ashley on 6/21/2017.
 */
public class Main {
    public static void main(String[] args) {
        if(args.length < 3) {
            System.out.println("Usage: spark-submit ClusteringTest.jar <path/input.jpg> <path/output.jpg> <k> ");
            return;
        }

        // Get image
        File file = new File(args[0]);
        BufferedImage image;
        try {
            image = ImageIO.read(file);
        } catch(IOException e) {
            System.err.println("Could not read file");
            return;
        }

        // Get RGB values of pixels
        int width = image.getWidth(), height = image.getHeight();
        int[][] pixels = new int[width][height];
        ArrayList<Integer> RGBdata = new ArrayList<Integer>(width * height);
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                pixels[i][j] = image.getRGB(i, j);
                RGBdata.add(image.getRGB(i, j));
            }
        }

        // Parse RGB data
        SparkConf conf = new SparkConf().setAppName("Image Cluster").setMaster("local");
        conf.setJars(new String[]{"C:\\Users\\kenneth.ashley\\IdeaProjects\\ClusteringTest\\out\\artifacts\\ClusteringTest_jar\\ClusteringTest.jar"});
        JavaSparkContext jsc = new JavaSparkContext(conf);
        JavaRDD<Integer> data = jsc.parallelize(RGBdata);
        JavaRDD<Vector> parsedData = data.map(
                new Function<Integer, Vector>() {
                    public Vector call(Integer i) {
                        int  red   = (i & 0x00ff0000) >> 16;
                        int  green = (i & 0x0000ff00) >> 8;
                        int  blue  =  i & 0x000000ff;
                        return Vectors.dense(new double[]{red, green, blue});
                    }
                }
        );
        parsedData.cache();

        // Perform k-means clustering
        int k = Integer.parseInt(args[2]), maxIterations = 20;
        KMeansModel clusters = KMeans.train(parsedData.rdd(), k, maxIterations);

        // Get array containing cluster centers
        int[][] centersArray = new int[k][3];
        int c = 0;
        for (Vector center : clusters.clusterCenters()) {
            double[] rawCenter = center.toArray();
            for(int i = 0; i < centersArray[c].length; i++) {
                centersArray[c][i] = (int) (rawCenter[i] + 0.5);
            }
            ++c;
        }

        // Set new RGB values for output image
        BufferedImage newImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                int RGBvalue = getCluster(centersArray, pixels[i][j]);
                newImage.setRGB(i, j, RGBvalue);
            }
        }

        // Write new image
        File outputFile = new File(args[1]);
        try {
            ImageIO.write(newImage, "jpg", outputFile);
        } catch (IOException e1) {
            System.err.println("Could not write file");
            return;
        }
    }

    // Get corresponding cluster value for a given RGB value
    private static int getCluster(int[][] centers, int RGBvalue) {
        int  red   = (RGBvalue & 0x00ff0000) >> 16;
        int  green = (RGBvalue & 0x0000ff00) >> 8;
        int  blue  =  RGBvalue & 0x000000ff;

        int closestCenter = 0;
        double centerDistance = Math.sqrt(Math.pow((centers[0][0] - red), 2) + Math.pow((centers[0][1] - green), 2) + Math.pow((centers[0][2] - blue), 2));
        for(int i = 1; i < centers.length; i++) {
            double thisDistance = Math.sqrt(Math.pow((centers[i][0] - red), 2) + Math.pow((centers[i][1] - green), 2) + Math.pow((centers[i][2] - blue), 2));
            if(thisDistance < centerDistance) {
                centerDistance = thisDistance;
                closestCenter = i;
            }
        }

        int[] centerRGB = centers[closestCenter];
        return (centerRGB[0] << 16) + (centerRGB[1] << 8) + centerRGB[2];
    }
}
