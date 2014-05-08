import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
 
public class ReducerRegressionPegasos extends Reducer<Text, Text, NullWritable, Text> {
	MultipleOutputs<Text, Text> mos;
	
	@Override
	public void setup(Context context) {
	    mos = new MultipleOutputs(context);
	}
    
 @Override
 public void reduce(Text key, Iterable<Text> values, Context context)
            throws IOException, InterruptedException {
	 
	 /* get the features dimension, make few definitions/initializations */
	 Configuration conf = context.getConfiguration();
	 int first_tab, p = conf.getInt("p", 100);
	 Vector x, gamma_j = new Vector(p);
	 String str;
	 double sum_error = 0.0;
	 
	 if(key.toString().equals("mean error")) {
		 /* loop over all errors and take the sum */
		 for(Text v : values){
			 sum_error += Double.parseDouble(v.toString());
		 }
		 /* emit in the dedicated file for error sum */
		 mos.write("error", NullWritable.get(), new Text(String.valueOf(sum_error)));
	 }
	 else {
		 /* loop over all couples (y, x) and increment gamma_j) */
		 for(Text v : values){
			 str = v.toString();
			 first_tab = str.indexOf("\t"); 
			 x = new Vector(str.substring(first_tab + 1));
			 gamma_j = gamma_j.plus(x);
		 } 
		 /* emit gamma_j - there is one gamma_j per reducer - all gamma_j will be summed in the driver in order to build gamma_t */
		 context.write(NullWritable.get(), new Text(gamma_j.toString()));
	 }
 }
 
 @Override
 protected void cleanup(Context context) throws IOException, InterruptedException {
     mos.close();
 }
 
}