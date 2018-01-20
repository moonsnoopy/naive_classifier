package r36054016_ML_term_project;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

public class main { //data preprocessing for Glass

	private static final String FILENAME = "D:\\ºÓ¯Z\\¾÷¾¹¾Ç²ß\\dataset\\glass\\glass.txt";
	public static List<ArrayList<Integer>> RANDOMRESULT = new ArrayList<ArrayList<Integer>>();
	public static String[][] DISCRET_DATA=null;
	public static int[][] CjINEACHFOLD =null;
	public static int[][] CVFOLDSTATISITIC=null;
	public static int[][][][] FOLDStatistic = null;
	public static void main(String[] args) {
		int numData ,numAttr ;
		/*numData = Integer.valueOf(args[0]);
		numAttr = Integer.valueOf(args[1]);*/
		numData = 214;
		numAttr = 11;//include id and class value
		String[][] data = new String[numData][numAttr];
		try (BufferedReader br = new BufferedReader(new FileReader(FILENAME))) {
			String sCurrentLine;
			int i=0;
			while ((sCurrentLine = br.readLine()) != null) {
				String[] line = sCurrentLine.split(",");
				for(int j=0;j<numAttr;j++){
					data[i][j] = line[j];
				}
				i++;
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		DISCRET_DATA = Discretization(data);
		/*for (int i=0 ; i < numData; i++){
			System.out.println(Arrays.toString(DISCRET_DATA[i]));
		}
		int[] numCV = new int[]{0,0,0,0,0,0,0};*/
		int[][][][] foldSummary = null;
		foldSummary = Fivefold_summary(numData,numAttr);
		FOLDStatistic = new int[5][numAttr][10][foldSummary[0][0][0].length];
		SummaryToStatistic(foldSummary,numAttr);
		int[] attrOrder = null;
		attrOrder = SNB(foldSummary, numAttr);
		System.out.println("SNB result:"+Arrays.toString(attrOrder));
		Dirichlet(attrOrder);
	}
	//--------------------------------- dirichlet step1-------------------------------
	private static void Dirichlet(int[] attrOrder) {
		// TODO Auto-generated method stub
		int attrN = attrOrder.length;
		int[] distinct_CV={1,2,3,4,5,6,7};
		double[][] attrAlpha = new double[attrN][11];
		for (int i = 0; i < attrN; i++ ){
			for (int j = 0; j < 11; j++){
				attrAlpha[i][j]=1;
			}
		}
		for (int i : attrOrder){
			//System.out.println(i);
			System.out.println("attr"+(i+2)+": "+Arrays.toString(attrAlpha[i]));
			int arcMaxAlpha=1111;
			double  maxAccu =0.0;
			for (int alpha = 1; alpha<=60; alpha++){
				for (int j = 0; j < 11; j++){
					attrAlpha[i][j]=alpha;
				}
				double accu = FiveFoldDirichlet(attrAlpha);
				if (accu >maxAccu){
					maxAccu = accu;
					arcMaxAlpha = alpha;
				}
			}
			for (int j = 0; j < 11; j++){
				attrAlpha[i][j]=arcMaxAlpha;
			}
			System.out.println(Arrays.toString(attrAlpha[i]));
			System.out.println("     Accu:"+maxAccu);
		}
	}
	//------------------------------- dirichlet step2--------------------------------
	private static double FiveFoldDirichlet(double[][] attrAlpha) {
		double turePredict=0.0, num_test=0.0;
		int[] distinct_CV={1,2,3,4,5,6,7};
		int numCV = distinct_CV.length;
		for(int f=0;f<5;f++){
			for(int index : RANDOMRESULT.get(f)){
				num_test++;
				double maxProb = 0.0;
				int prediction = 1000;
				for(int cv = 0; cv<numCV;cv++){
					double Nj =CVFOLDSTATISITIC[f][cv],N=DISCRET_DATA.length-RANDOMRESULT.get(f).size();
					double prob = Nj/N;
					for(int a = 0; a < attrAlpha.length; a++){
						int bin = Integer.valueOf(DISCRET_DATA[index][a+1]);
						double alpha = 0.0;
						for ( int aI = 0;aI < attrAlpha[a].length;aI++){
							alpha += attrAlpha[a][aI];
						}
						double Yij = FOLDStatistic[f][a][bin][cv]; // i=bin j=cv
						prob =prob * (Yij+attrAlpha[a][bin])/(Nj+alpha);

					}
					if(prob > maxProb){
						maxProb = prob;
						prediction = cv+1;
					}
				}
				//System.out.println("predict:"+prediction+" true:"+DISCRET_DATA[index][numAttr-1]+" "+Arrays.toString(probability));
				if(prediction == Integer.valueOf(DISCRET_DATA[index][attrAlpha.length+1])){
					turePredict++;
				}
			}
		}
		double avgAccuracy = turePredict/num_test;
		return avgAccuracy;
	}
	//--------------------------------  SNB --------------------------------------
	private static int[] SNB(int[][][][] foldSummary, int numAttr) {
		int[] attrOrder= new int[numAttr-2];
		for(int i=0; i<numAttr-2;i++){
			attrOrder[i]=111111;
		}
		for(int i = 0; i < numAttr-2; i++){
			double maxAccu = 0.0;
			int arcMaxAccu=111111;
			for (int a =0; a< numAttr-2;a++){
				//System.out.println(a+" "+Arrays.toString(attrOrder)+" TF:"+isExist(attrOrder,a));
				if ( !(isExist(attrOrder,a)) ){
					double accuracy = FiveFoldNB(foldSummary, a, attrOrder,numAttr);
					//System.out.println(accuracy);
					if(accuracy>maxAccu){
						maxAccu = accuracy;
						arcMaxAccu = a;
					}
					//System.out.println(String.valueOf(maxAccu)+" "+ arcMaxAccu);
				}
			}
			attrOrder[i] = arcMaxAccu;
		}
		return attrOrder;
	}
	//------------------------------- check whether Array contain a certain value ----------------
	public static boolean isExist(int[] arr, int targetValue) {
		for(int s: arr){
			if(s ==targetValue)
				return true;
		}
		return false;
	}
	//--------------------------------  Five Fold Naive Bayesian --------------------------------------
	private static double FiveFoldNB(int[][][][] foldSummary, int testAttr, int[] attrOrder,int numAttr) {
		double turePredict=0.0, num_test=0.0;
		int[] distinct_CV={1,2,3,4,5,6,7};
		int numCV = distinct_CV.length;
		for(int f=0;f<5;f++){
			for(int index : RANDOMRESULT.get(f)){
				num_test++;
				double maxProb = 0.0;
				int prediction = 1000;
				double[] probability = new double[numCV];
				for(int cv = 0; cv<numCV;cv++){
					double Nj =CVFOLDSTATISITIC[f][cv],N=DISCRET_DATA.length-RANDOMRESULT.get(f).size();
					double prob = Nj/N;
					for(int a = 0; a < numAttr-1; a++){
						if(a == testAttr || Arrays.asList(attrOrder).contains(a)){
							int bin = Integer.valueOf(DISCRET_DATA[index][a+1]);
							double Nij = FOLDStatistic[f][a][bin][cv];
							prob =prob * (Nij+1)/(Nj+10);
						}
					}
					probability[cv]=prob;
					if(prob > maxProb){
						maxProb = prob;
						prediction = cv+1;
					}
				}
				//System.out.println("predict:"+prediction+" true:"+DISCRET_DATA[index][numAttr-1]+" "+Arrays.toString(probability));
				if(prediction == Integer.valueOf(DISCRET_DATA[index][numAttr-1])){
					turePredict++;
				}
			}
		}
		double avgAccuracy = turePredict/num_test;
		return avgAccuracy;
	}
	//--------------------------------  calculate Nj and Nij for each 4fold--------------------------------------
		private static void SummaryToStatistic(int[][][][] foldSummary,int numAttr) {
			// TODO Auto-generated method stub
			int numCV = foldSummary[0][0][0].length;
			CVFOLDSTATISITIC =new int[5][numCV];
			for(int f= 0;f<5;f++){
		    	for(int cv=0; cv< numCV; cv++){
		    		int Nj=0;
					for(int f2=0;f2<5;f2++){
						if(f2!=f){
							Nj += CjINEACHFOLD[f2][cv];
						}
					}
					CVFOLDSTATISITIC[f][cv]=Nj;
		    		for(int a = 1; a <numAttr - 1; a++){
		    			for(int b = 0; b < 10; b++){
		    				int Nij_training=0;
		    				for(int f2=0;f2<5;f2++){
		    					if(f2!=f){
		    						Nij_training+=foldSummary[f2][a][b][cv];
		    					}
		    				}
		    				FOLDStatistic[f][a][b][cv]=Nij_training;
		    			}
		    		}
		    	}
		    }
		}
	//--------------------------------  divide into k fold and calculate the amount --------------------------------------
	private static int[][][][] Fivefold_summary(int numData, int numAttr) {
		// TODO Auto-generated method stub
		int[] distinct_CV={1,2,3,4,5,6,7};
		int k = 5;
		Random random = new Random();
	    int result[] = new int[numData];
	    for (int i = 0 ; i < numData; i++){
	    	result[i] = i;
	    }
	    for(int i=0; i < result.length; i ++){
	        int index = random.nextInt(numData);
	        int tmp = result[index];
	        result[index] = result[i];
	        result[i] = tmp;
	      }
	    //System.out.println("result: "+ Arrays.toString(result));
	    int[][][][] summary = new int[k][numAttr-1][10][distinct_CV.length];
	    int foldPT=0;
	    for(int f= 0;f<k;f++){
	    	for(int i = 0; i <numAttr - 1; i++){
	    		for(int j = 0; j < 10; j++){
	    			for(int c = 0; c < distinct_CV.length; c++){
	    				summary[f][i][j][c]=0;
	    			}
	    		}
	    	}
	    }
	    ArrayList<Integer> fold0List = new ArrayList<Integer>();
	    ArrayList<Integer> fold1List = new ArrayList<Integer>();
	    ArrayList<Integer> fold2List = new ArrayList<Integer>();
	    ArrayList<Integer> fold3List = new ArrayList<Integer>();
	    ArrayList<Integer> fold4List = new ArrayList<Integer>();
	    int[][] numOfCVInFold = new int[5][distinct_CV.length];  //NUM_EACH_CV[cv][fold]
	    for (int i =0;i< distinct_CV.length;i++){
	    	for (int j = 0; j < 5;j++){
	    		numOfCVInFold[j][i] = 0;
	    	}
	    }
	    for (int r = 0; r < result.length; r++){
	    	foldPT = (int) Math.floor(r/Math.ceil((double)numData/k));
	    	//System.out.println("fold " + foldPT+ ":"+Arrays.toString(DISCRET_DATA[result[r]]));
	    	switch (foldPT){
	    	case 0: fold0List.add(result[r]);
	    			break;
	    	case 1: fold1List.add(result[r]);
			 		break;
	    	case 2: fold2List.add(result[r]);
			 		break;
	    	case 3: fold3List.add(result[r]);
			 		break;
	    	case 4: fold4List.add(result[r]);
			 		break;
	    	}
	    	int cv = Integer.valueOf(DISCRET_DATA[result[r]][numAttr-1]);  // find which class value
	    	for(int j = 1; j <numAttr - 1; j++){
	    		int bin = Integer.valueOf(DISCRET_DATA[result[r]][j]);
	    		//System.out.println("fold " + foldPT+ " attr "+ (j) +" bin "+bin+" CV "+ (cv-1));
	    		summary[foldPT][j-1][bin][cv-1]++;
	    	}
	    	numOfCVInFold[foldPT][cv-1]++;
	    }
	    RANDOMRESULT.add(fold0List);
    	RANDOMRESULT.add(fold1List);
    	RANDOMRESULT.add(fold2List);
    	RANDOMRESULT.add(fold3List);
    	RANDOMRESULT.add(fold4List);
	    //System.out.println("New result: "+Arrays.toString(RANDOMRESULT.toArray()));
	    CjINEACHFOLD =numOfCVInFold;
	    return summary;
	}

	//--------------------------------  Discretization -------------------------------------------------
	public static String[][] Discretization(String[][] rawData){
		int numAttr = rawData[0].length;
		int numdata = rawData.length;
		double[][] maxMin = new double[numAttr][2];  // maxmin[][0] = max maxMin[][1] = min
		for(int j = 1 ; j < numAttr-1 ; j ++){ // find the range of each attr
			maxMin[j][0]= 0;
			maxMin[j][1] = 1000;
			for (int i = 0 ; i < numdata; i++){
				if (Double.valueOf(rawData[i][j]) > maxMin[j][0]){
					maxMin[j][0] = Double.valueOf(rawData[i][j]);
				}
				if (Double.valueOf(rawData[i][j]) < maxMin[j][1]){
					maxMin[j][1] = Double.valueOf(rawData[i][j]);
				}
			}
		}
		for (int j =1; j < numAttr-1; j++){
			//System.out.println(Integer.toString(j)+" max: "+ Double.toString(maxMin[j][0])+" min: "+ Double.toString(maxMin[j][1]) );
			for (int i=0 ; i < numdata; i++){
				double interval = (maxMin[j][0] - maxMin[j][1])/10;
				rawData[i][j] = String.valueOf((Double.valueOf(rawData[i][j])-Double.valueOf(maxMin[j][1])) / interval);
				if(Double.valueOf(rawData[i][j])==10.0){ rawData[i][j]="9";}
				rawData[i][j]=String.valueOf(rawData[i][j].charAt(0));
			}
		}


		return rawData;
	}
}

