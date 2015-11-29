/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package classifier.neuralnetwork;

import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

/**
 *
 * @author asus
 */
public class DeltaIncrementalCoba {
     public static void main(String[] args){
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
        
        Scanner reader = new Scanner(System.in);
        
        System.out.println("Max epoch: ");
        maxEpoch = reader.nextInt();
        
        System.out.println("Learning rate: ");
        learningRate = reader.nextDouble();
        
        System.out.println("Momentum: ");
        momentum = reader.nextDouble();
        
        System.out.println("Jumlah input: ");
        juminput = reader.nextInt();
        
        int jumtarget;
        List<Double> targetList = new ArrayList<>();
        
        System.out.println("Jumlah target: ");
        jumtarget = reader.nextInt();
        
        for(int j=0;j<juminput*jumtarget;j++){
            inputList.add((double)0);
        }
        
        for(int i=0;i<jumtarget;i++){
            System.out.println("Target "+(i+1)+" : ");
            input = reader.nextDouble();
            targetList.add(input);
        }
        
         for(int i=0;i<jumtarget;i++){
            errorListEpoch.add((double)0);
        }
         
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
            for(int i=0;i<targetList.size();i++){
                sum = 0;
                System.out.println("epoch="+(h+1));
                System.out.println("iteration="+(i+1));
                if(h==0){
                    for(int j=i*juminput;j<(i+1)*juminput;j++){
                        System.out.println("Input " + j + " : ");
                        input = reader.nextDouble();
                        inputList.set(j, input);
                    }
                }
                //Hitung Sum
                for(int j=i*juminput;j<(i+1)*juminput;j++){
                    System.out.println("neuron="+neuron.getCurrentWeight().get(j%4));
                    sum = sum + (inputList.get(j)*neuron.getCurrentWeight().get(j%4));
                }
                System.out.println("sum="+sum);
                error = targetList.get(i) - sum;
                errorListEpoch.set(i, error);
                System.out.println("targetlist="+targetList.get(i));
                System.out.println("error="+error);
                for(int j=0;j<juminput;j++){
                    System.out.println("SetNewDeltaW"+j+"= "+inputList.get(j)+"*"+error+"*"+"learningRate");
                    neuron.setDeltaWeight(j, (inputList.get(j+(i*juminput))*error*learningRate)+(momentum*neuron.getPreviousDeltaWeight().get(j)));
                    neuron.setCurrentWeight(j, neuron.getCurrentWeight().get(j)+neuron.getDeltaWeight().get(j));
                    //System.out.println("new w"+j+"= "+neuron.getCurrentWeight().get(j));
                    System.out.println("new w"+j+"= "+neuron.getCurrentWeight().get(j));
                    neuron.setPreviousDeltaWeight(j, neuron.getDeltaWeight().get(j));
                }
            }
            double tempSum = 0;
            double tempSumError = 0;
            for(int i=0;i<jumtarget;i++){
//                tempSum = 0;
//                for(int j=0;j<juminput;j++){
//                    tempSum = tempSum + inputList.get(j+(i*juminput));
//                }
////                errorSquareIterationAfterEpoch.set(i, Math.pow(targetList.get(i)-signFunction(tempSum),2));
//                tempSumError = tempSumError + Math.pow(targetList.get(i)-tempSum,2);
//                errorEpoch = (int) (tempSumError * 0.5);
                tempSum = tempSum + Math.pow(errorListEpoch.get(i), 2);
            }
            errorEpoch = tempSum / 2;
            System.out.println("errorEpoch = "+errorEpoch);
            h++;
            System.out.println("==========End of Epoch==========");
        }while(h<=maxEpoch && errorEpoch!=0);
    }
    
    public static int signFunction(double sum){
        int signResult = 0;
        if(sum>=0){
            signResult = 1;
        } else {
            signResult = -1;
        }
        return signResult;
    } 
}
