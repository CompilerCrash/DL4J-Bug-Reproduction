import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.writable.IntWritable;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;

import java.util.Collections;

public class Mmul {

    private static final INDArray a = Nd4j.ones(2, 6);
    private static final INDArray b = Nd4j.ones(6, 4);

    public static void main(String[] args) {
        String separator = "=".repeat(60);

        nd4jMmul();
        System.out.println(separator);
        nd4jGemm();
        System.out.println(separator);
        nd4jBatchMmul(); // bug here
        System.out.println(separator);
        nd4jTensorMmul();
        System.out.println(separator);
        sdMmul();
        System.out.println(separator);
        sdMatmul();
        System.out.println(separator);
        sdBatchMmul(); // bug here
        System.out.println(separator);
        sdBatchMmulArray(); // bug here
        System.out.println(separator);
        sdTensorMmul();
        System.out.println(separator);
        sdTensorMmulBackprop(); // bug here
    }

    public static void nd4jMmul() {
        INDArray res = Nd4j.base.mmul(a, b);
        System.out.println(res); // correct
    }

    public static void nd4jGemm() {
        INDArray res = Nd4j.gemm(a, b, false, false);
        System.out.println(res); // correct
    }

    public static void nd4jBatchMmul() {
        INDArray[] res = Nd4j.base.batchMmul(new INDArray[]{a}, b); // throws exception
        System.out.println(res[0]);
    }

    public static void nd4jTensorMmul() {
        INDArray res = Nd4j.tensorMmul(a, b, new int[][]{{1}, {0}});
        System.out.println(res); // correct
    }

    public static void sdMmul() {
        SameDiff sd = SameDiff.create();
        SDVariable res = sd.mmul(sd.constant(a), sd.constant(b));
        System.out.println(res.eval()); // correct
    }

    public static void sdMatmul() {
        SameDiff sd = SameDiff.create();
        SDVariable res = sd.linalg.matmul(sd.expandDims(sd.constant(a), 0), sd.expandDims(sd.constant(b), 0));
        System.out.println(res.eval()); // correct
    }

    public static void sdBatchMmul() {
        SameDiff sd = SameDiff.create();
        SDVariable[] res = sd.batchMmul(new SDVariable[]{sd.constant(a)}, sd.constant(b));
        System.out.println(res[0].eval()); // wrong result or crash
    }

    public static void sdBatchMmulArray() {
        SameDiff sd = SameDiff.create();
        SDVariable[] res = sd.batchMmul(new SDVariable[]{sd.constant(a).add(0)}, sd.constant(b).add(0)); // throws exception
        System.out.println(res[0].eval());
    }

    public static void sdTensorMmul() {
        SameDiff sd = SameDiff.create();
        SDVariable res = sd.tensorMmul(sd.constant(a), sd.constant(b), new int[]{1}, 0);
        System.out.println(res.eval()); // correct
    }

    public static void sdTensorMmulBackprop() {
        int batchSize = 4;
        int seqLength = 8;

        SameDiff sd = SameDiff.create();

        SDVariable features = sd.placeHolder("features", DataType.FLOAT, batchSize, seqLength);
        SDVariable labels = sd.placeHolder("labels", DataType.FLOAT, batchSize, batchSize);
        SDVariable var = sd.var("variable", seqLength, batchSize);
        SDVariable predictions = sd.tensorMmul("predictions", features, var, new int[]{1}, 0);
        sd.loss.meanSquaredError("loss", labels, predictions, null);

        TrainingConfig config = new TrainingConfig.Builder()
                .updater(new Adam(0.1))
                .dataSetFeatureMapping("features")
                .dataSetLabelMapping("labels")
                .build();
        sd.setTrainingConfig(config);

        RecordReader reader = new CollectionRecordReader(
                Collections.nCopies(batchSize, Collections.nCopies(seqLength + batchSize, new IntWritable(1))));
        DataSetIterator iterator = new RecordReaderDataSetIterator(
                reader, batchSize, seqLength, seqLength + batchSize - 1, true);

        System.out.println(sd.output(iterator, "predictions").get("predictions")); // works

        sd.fit(iterator, 1); // throws exception
    }
}
