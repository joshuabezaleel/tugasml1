/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package classifier.neuralnetwork;

import static classifier.neuralnetwork.MyPTR.signFunction;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;

/**
 *
 * @author asus
 */
public class MyDeltaBatch extends Classifier {
        Neuron neuron = new Neuron();
        List<Double> inputList = new ArrayList<>();
        List<Double> target = new ArrayList<>();
        List<Double> errorListEpoch = new ArrayList<>();
        int juminput;
        double input;
        int currEpoch = 1;
        int maxEpoch;
        double error;
        double learningRate;
        double momentum;
        
    public MyDeltaBatch(){
        maxEpoch = 10;
        learningRate = 0.1;
        momentum = 0.0;
    }    

    @Override
    public void buildClassifier(Instances data) throws Exception {
        NominalToBinary nomToBinFilter = new NominalToBinary();
        Normalize normalizeFilter = new Normalize();
        
        // PREPARE DATASET
        data = new Instances(data);
        data.deleteWithMissingClass();
        nomToBinFilter.setInputFormat(data);
        data = Filter.useFilter(data, nomToBinFilter);
        normalizeFilter.setInputFormat(data);
        data = Filter.useFilter(data, normalizeFilter);
        
        Scanner reader = new Scanner(System.in);
        
        juminput = data.numAttributes()-1;
        
        List<Double> targetList = new ArrayList<>();
        
        for(int j=0;j<(data.numAttributes()-1)*data.numInstances();j++){
            inputList.add((double)0);
        }
        
//        for(int i=0;i<jumtarget;i++){
//            System.out.println("Target "+(i+1)+" : ");
//            input = reader.nextDouble();
//            targetList.add(input);
//        }
        
        for(int i=0;i<data.numInstances();i++){
            errorListEpoch.add((double)0);
        }
        
        juminput = data.numAttributes()-1;
        neuron.initPreviousWeight(juminput,0);
        neuron.initCurrentWeight(juminput,0);
        neuron.initDeltaWeight(juminput,0);
        neuron.initPreviousDeltaWeight(juminput,0);
        //neuron.randomizeCurrentWeight();
        neuron.setCurrentWeightZero();
        
        double sum;
        
        double errorEpoch = 0;
        
        int iteration;
        int h=0;
        do{
            //1 Epoch
            for(int i=0;i<data.numInstances();i++){
                sum = 0;
                System.out.println("epoch="+(h+1));
                System.out.println("iteration="+(i+1));
                if(h==0){
                    //Inisialisasi bias
                    for(int j=0;j<data.numInstances();j++){
                        inputList.add(j*(data.numAttributes()-1), (double)1);
                    }
                    //Input attribute value to inputList
                    for(int j=0;j<data.numInstances();j++){
                        for(int k=0;k<data.numAttributes()-1;k++){
                            inputList.add((k+1)*j*(data.numAttributes()-1),data.instance(j+1).value(k+1));
                        }
                    }
                }
                //Hitung sum
                for(int j=i*juminput;j<(i+1)*juminput;j++){
                    sum = sum + (inputList.get(j)*neuron.getCurrentWeight().get(j%4));
                }
                System.out.println("//sum = "+sum);
                error = targetList.get(i) - sum;
                errorListEpoch.set(i, error);
                System.out.println("//targetlist = "+targetList.get(i));
                System.out.println("//error="+error);
            }
            //Penghitungan DeltaWeight dan NewWeight
            for(int j=0;j<juminput;j++){
                double tempDelta = 0;
                for(int k=0;k<data.numInstances();k++){
//                    System.out.println("inputList ke-"+(k*juminput+j)+"= "+inputList.get(k*juminput+j));
//                    System.out.println("error ke-"+j+"= "+errorListEpoch.get(k));
                    tempDelta = tempDelta + (inputList.get(k*juminput+j)*errorListEpoch.get(k)*learningRate);
                }
                neuron.setDeltaWeight(j, tempDelta);
                System.out.println("DeltaWeightW"+j+"="+tempDelta);
                neuron.setCurrentWeight(j, neuron.getCurrentWeight().get(j)+neuron.getDeltaWeight().get(j));
                neuron.setPreviousDeltaWeight(j, neuron.getDeltaWeight().get(j));
                //System.out.println("SetNewDeltaW"+j+"= "+inputList.get(j)+"*"+error+"*"+"learningRate");
                //neuron.setDeltaWeight(j, (inputList.get(j+(i*juminput))*error*learningRate)+(momentum*neuron.getPreviousDeltaWeight().get(j)));
                //neuron.setCurrentWeight(j, neuron.getCurrentWeight().get(j)+neuron.getDeltaWeight().get(j));
                //System.out.println("new w"+j+"= "+neuron.getCurrentWeight().get(j));
                //System.out.println("new w"+j+"= "+neuron.getCurrentWeight().get(j));
                //neuron.setPreviousDeltaWeight(j, neuron.getDeltaWeight().get(j));
            }
            double tempSum = 0;
            double tempSumError = 0;
            for(int i=0;i<data.numInstances();i++){
                  tempSum = tempSum + Math.pow(errorListEpoch.get(i), 2);
//                tempSum = 0;
//                for(int j=0;j<juminput;j++){
//                    tempSum = tempSum + inputList.get(j+(i*juminput));
//                }
//                tempSumError = tempSumError + Math.pow(targetList.get(i)-signFunction(tempSum),2);
//                errorEpoch = (int) (tempSumError * 0.5);
            }
            errorEpoch = tempSum / 2;
            System.out.println("errorEpoch = "+errorEpoch);
            h++;
            System.out.println("===========End of Epoch===========");
        }while(h<=maxEpoch && errorEpoch!=0);
    }
    
    @Override
    public double classifyInstance(Instance data) {
        List<Double> attributeValue = new ArrayList<>();
        //Inisialisasi bias
        attributeValue.add((double)1);
        
        //Input attribute value to inputList
        for(int k=0;k<data.numAttributes()-1;k++){
            inputList.add(data.value(k));
        }
        
        //including bias
        double tempSum = 0;
        for(int k=0;k<data.numAttributes();k++){
            tempSum = tempSum + (inputList.get(k)*neuron.getCurrentWeight().get(k));
        }
        
        return signFunction(tempSum);
    }
    
}
