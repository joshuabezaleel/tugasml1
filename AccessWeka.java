import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.Random;
import java.util.Scanner;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.neural.LinearUnit;
import weka.classifiers.functions.neural.SigmoidUnit;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instances;

public class TugasAI
{
	//Attributes
	private Instances data;
	private Instances labeleddata;
	private Classifier classifier;
	private Random rand;
	//Methods
	public Instances getLabeledData()
	{
		return labeleddata;
	}
	public void bacaFile(String path) throws Exception
	{
		DataSource source = new DataSource(path);
		data = source.getDataSet();
		if(data.classIndex()==-1)
			data.setClassIndex(data.numAttributes()-1);
	}
	public void buildJ48Classifier() throws Exception
	{
		classifier = new J48();
		classifier.buildClassifier(data);
	}
	public void buildBayesClassifier() throws Exception
	{
		classifier = new NaiveBayes();
		classifier.buildClassifier(data);
	}
	public void buildKNNClassifier() throws Exception
	{
		classifier = new IBk();
		classifier.buildClassifier(data);
	}
	public void buildANNClassifier() throws Exception
	{
		classifier = new MultilayerPerceptron();
		classifier.buildClassifier(data);
		
	}
	public void LabelData(String unlabelPath) throws Exception
	{
		DataSource source = new DataSource(unlabelPath);
		data = source.getDataSet();
		if(data.classIndex()==-1)
			data.setClassIndex(data.numAttributes()-1);
		
		labeleddata = new Instances(data);
		for(int i=0;i< data.numInstances(); i++)
		{
			double clsLabel = classifier.classifyInstance(data.instance(i));
			labeleddata.instance(i).setClassValue(clsLabel);
		}
		BufferedWriter output = new BufferedWriter(new FileWriter("output.arff"));
		output.write(labeleddata.toString());
		System.out.println(labeleddata.toString());
		output.newLine();
		output.flush();
		output.close();
	}
	public void BuildModel(String outputfile) throws Exception
	{
		weka.core.SerializationHelper.write(outputfile, classifier);
	}
	public void evaluate() throws Exception
	{
		Evaluation eval = new Evaluation(data);
		eval.evaluateModel(classifier, data); //use training set
		System.out.println(eval.toSummaryString());
	}
	public void evalCross(int folds) throws Exception
	{
		Evaluation eval = new Evaluation(data);
		rand = new Random(1); // cross validation
		eval.crossValidateModel(classifier, data, folds, rand); //cross validation 10
		System.out.println(eval.toSummaryString());
	}
	
	//main
	public static void main(String[] args){
		TugasAI dt = new TugasAI();
		System.out.println("Pilih classifier");
		System.out.println("1. NaiveBayes");
		System.out.println("2. J48");
		System.out.println("3. KNN");
		System.out.println("4. ANN");
		Scanner keyboard = new Scanner(System.in);
		int pil = keyboard.nextInt();
		System.out.println("Test options");
		System.out.println("1. Training Model");
		System.out.println("2. Cross Validation");
		int mod = keyboard.nextInt();
		int folds = 0;
		if(mod==2){
			System.out.println("masukkan jumlah fold : ");
			folds = keyboard.nextInt();
		}
		try{
    			dt.bacaFile("weather.nominal.arff");
			switch(pil){
			case 1:
				dt.buildBayesClassifier();
				if(mod == 1) dt.evaluate();
				else dt.evalCross(folds);
				dt.BuildModel("NaiveBayes.model");
				break;
			case 2:
				dt.buildJ48Classifier();
				if(mod == 1) dt.evaluate();
				else dt.evalCross(folds);
				dt.BuildModel("J48.model");
				break;
			case 3:
				dt.buildKNNClassifier();
				if(mod == 1) dt.evaluate();
				else dt.evalCross(folds);
				dt.BuildModel("KNN.model");
				break;
			case 4:
				dt.buildANNClassifier();
				if(mod == 1) dt.evaluate();
				else dt.evalCross(folds);
				dt.BuildModel("ANN.model");
				break;
			}
			dt.LabelData("weatherunlabeled.arff");
//			System.out.println(dt.getLabeledData().toString());
			//dt.buildANNLinearClassifier();
		}
		catch (Exception e){
			System.out.println(e.getMessage());
		}
	}
}
