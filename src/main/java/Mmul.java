import org.deeplearning4j.datasets.iterator.RandomDataSetIterator;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;

import static org.deeplearning4j.datasets.iterator.RandomDataSetIterator.Values.ONES;

public class Mmul {

    private static final INDArray a = Nd4j.ones(2, 6);
    private static final INDArray b = Nd4j.ones(6, 4);
    private static final INDArray alpha = Nd4j.scalar(1f);
    private static final INDArray beta = Nd4j.scalar(0f);

    public static void main(String[] args) {
        String separator = "=".repeat(60);

        nd4jMmul();
        System.out.println(separator);
        nd4jGemm();
        System.out.println(separator);
        nd4jBatchMmul(); // bug here
        System.out.println(separator);
        nd4jBatchMmulInt();
        System.out.println(separator);
        nd4jBatchMmulDouble(); // bug here
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
        sdBatchMmulInt();
        System.out.println(separator);
        sdBatchMmulDouble();
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
        INDArray[] res = Nd4j.base.batchMmul(alpha, beta, new INDArray[]{a}, b); // throws exception
        System.out.println(res[0]);
    }

    public static void nd4jBatchMmulInt() {
        try {
            INDArray[] res = Nd4j.base.batchMmul(alpha.castTo(DataType.INT32), beta.castTo(DataType.INT32), new INDArray[]{a}, b);
            System.out.println(res[0]);
        } catch (ND4JIllegalStateException e) {
            e.printStackTrace(); // throws exception as it should (but a bit cryptic)
        }
    }

    // FIXME
    public static void nd4jBatchMmulDouble() {
        INDArray[] res = Nd4j.base.batchMmul(alpha.castTo(DataType.DOUBLE), beta.castTo(DataType.DOUBLE), new INDArray[]{a}, b);
        System.out.println(res[0]); // wrong result
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
        SDVariable[] res = sd.batchMmul(sd.constant(alpha), sd.constant(beta), new SDVariable[]{sd.constant(a)}, sd.constant(b));
        System.out.println(res[0].eval()); // wrong result (or crash)
    }

    public static void sdBatchMmulArray() {
        SameDiff sd = SameDiff.create();
        SDVariable[] res = sd.batchMmul(sd.constant(alpha), sd.constant(beta), new SDVariable[]{sd.constant(a).add(0)}, sd.constant(b).add(0)); // throws exception
        System.out.println(res[0].eval());
    }

    public static void sdBatchMmulInt() {
        try {
            SameDiff sd = SameDiff.create();
            SDVariable[] res = sd.batchMmul(sd.constant(alpha).castTo(DataType.INT32), sd.constant(beta).castTo(DataType.INT32), new SDVariable[]{sd.constant(a)}, sd.constant(b));
            System.out.println(res[0].eval());
        } catch (IllegalStateException e) {
            e.printStackTrace(); // throws exception as it should
        }
    }

    // FIXME
    public static void sdBatchMmulDouble() {
        SameDiff sd = SameDiff.create();
        SDVariable[] res = sd.batchMmul(sd.constant(alpha).castTo(DataType.DOUBLE), sd.constant(beta).castTo(DataType.DOUBLE), new SDVariable[]{sd.constant(a)}, sd.constant(b));
        System.out.println(res[0].eval()); // wrong result
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

        DataSetIterator iterator = new RandomDataSetIterator(1, new long[]{batchSize, seqLength}, new long[]{batchSize, batchSize}, ONES, ONES);

        System.out.println(sd.output(iterator, "predictions").get("predictions")); // forward pass works

        sd.fit(iterator, 1); // backward pass throws exception
    }
}
