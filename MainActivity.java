package com.example.plantdiseasedetection;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;


import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.example.plantdiseasedetection.ml.PlantliteModel;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.text.DecimalFormat;

public class MainActivity extends AppCompatActivity {
    private static final int REQUEST_IMAGE_CAPTURE = 101;
    private static final int REQUEST_IMAGE_FROM_GALLERY = 102;
    private Button upload;
    private Button capture;
    private Button predict;
    private TextView result;
    private TextView result2;
    private TextView result3;
    private ImageView imagePlant;
    private Interpreter tfliteInterpreterlenet;
    private Interpreter tfliteInterpretervgg16;
    private Interpreter tfliteInterpreterdensenet;
    int ImgSize = 256;
    int numClasses = 15;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        // Defining the buttons and textviews and imageview
        upload = findViewById(R.id.upload);
        capture = findViewById(R.id.capture);
        predict = findViewById(R.id.predict);
        result = findViewById(R.id.result);
        result2 = findViewById(R.id.result1);
        result3 = findViewById(R.id.result2);
        imagePlant = findViewById(R.id.plantImage);

        try {
            // Loading the LeNet5 Model
            tfliteInterpreterlenet = new Interpreter(loadModelFilelenet());
        } catch (IOException e) {
            Toast.makeText(this, "Error loading model", Toast.LENGTH_SHORT).show();
            e.printStackTrace();
        }
        try {
            // Loading the VGG16 Model
            tfliteInterpretervgg16 = new Interpreter(loadModelFilevgg16());
        } catch (IOException e) {
            Toast.makeText(this, "Error loading model", Toast.LENGTH_SHORT).show();
            e.printStackTrace();
        }
        try {
            // Loading the DenseNet24 Model
            tfliteInterpreterdensenet = new Interpreter(loadModelFiledensenet());
        } catch (IOException e) {
            Toast.makeText(this, "Error loading model", Toast.LENGTH_SHORT).show();
            e.printStackTrace();
        }

        // Launching the camera if caputure button is clicked
        capture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent camera_intent = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(camera_intent, REQUEST_IMAGE_CAPTURE);
            }

        });
        // Launching the gallery files if Upload Button is clicked
        upload.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(intent, REQUEST_IMAGE_FROM_GALLERY);
            }
        });




    };

    private MappedByteBuffer loadModelFilelenet() throws IOException {
        // Open the lenet5 model file from the assets folder
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("lenet5new.tflite");
        // Creating a fileinput stream to read the
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        // Get the filechannel from the fileinputstream
        FileChannel fileChannel = inputStream.getChannel();
        // get the start offset of the model file within the assets folder
        long startOffset = fileDescriptor.getStartOffset();
        // gET the lenght of the model file
        long declaredLength = fileDescriptor.getDeclaredLength();
        // map the model file into memory and return as mappedbytebuffer
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
    private MappedByteBuffer loadModelFilevgg16() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("vgg16.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
    private MappedByteBuffer loadModelFiledensenet() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("Densenet.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public  void predictImage(Bitmap img) {

        try {
            // Resize input image to the required dimensions
            Bitmap resizedImage = Bitmap.createScaledBitmap(img, ImgSize, ImgSize, false);

            // Convert Bitmap to ByteBuffer
            ByteBuffer inputBuffer = ByteBuffer.allocateDirect(ImgSize * ImgSize * 3 * 4); // 4 bytes per float
            inputBuffer.order(ByteOrder.nativeOrder());
            int[] pixels = new int[ImgSize * ImgSize];
            resizedImage.getPixels(pixels, 0, ImgSize, 0, 0, ImgSize, ImgSize);
            for (int pixelValue : pixels) {
                inputBuffer.putFloat((Color.red(pixelValue) / 255.0f)); // NORMALIZING THE DATA
                inputBuffer.putFloat((Color.green(pixelValue) / 255.0f));
                inputBuffer.putFloat((Color.blue(pixelValue) / 255.0f));
            }
            Bitmap resizedImagevgg = Bitmap.createScaledBitmap(img, 64, 64, false);

            // Convert Bitmap to ByteBuffer
            ByteBuffer inputBuffervgg = ByteBuffer.allocateDirect(64 * 64 * 3 * 4); // 4 bytes per float
            inputBuffervgg.order(ByteOrder.nativeOrder());
            int[] pixelsvgg = new int[64 * 64];
            resizedImagevgg.getPixels(pixelsvgg, 0, 64, 0, 0, 64, 64);
            for (int pixelValuevgg : pixelsvgg) {
                inputBuffervgg.putFloat((Color.red(pixelValuevgg) / 255.0f));
                inputBuffervgg.putFloat((Color.green(pixelValuevgg) / 255.0f));
                inputBuffervgg.putFloat((Color.blue(pixelValuevgg) / 255.0f));
            }

            // Run THE MODEL AND INPUT THE IMAGE TO GET PREDICTIONS
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, ImgSize, ImgSize, 3}, DataType.FLOAT32);
            inputFeature0.loadBuffer(inputBuffer);
            TensorBuffer inputFeature0vgg = TensorBuffer.createFixedSize(new int[]{1, 64, 64, 3}, DataType.FLOAT32);
            inputFeature0vgg.loadBuffer(inputBuffervgg);

            TensorBuffer outputFeature0lenet = TensorBuffer.createFixedSize(new int[]{1, numClasses}, DataType.FLOAT32);
            TensorBuffer outputFeature0vgg16 = TensorBuffer.createFixedSize(new int[]{1, numClasses}, DataType.FLOAT32);
            TensorBuffer outputFeature0densenet = TensorBuffer.createFixedSize(new int[]{1, numClasses}, DataType.FLOAT32);

            tfliteInterpreterlenet.run(inputFeature0.getBuffer(), outputFeature0lenet.getBuffer().rewind());
            tfliteInterpretervgg16.run(inputFeature0vgg.getBuffer(), outputFeature0vgg16.getBuffer().rewind());
            tfliteInterpreterdensenet.run(inputFeature0.getBuffer(), outputFeature0densenet.getBuffer().rewind());

            // Find the class with the highest confidence
            int predictedClassIndexlenet = argmax(outputFeature0lenet.getFloatArray());
            int predictedClassIndexvgg16 = argmax(outputFeature0vgg16.getFloatArray());
            int predictedClassIndexdensenet = argmax(outputFeature0densenet.getFloatArray());

            // Display the predicted class
            String[] classes = {"Pepper__bell___Bacterial_spot", "Pepper__bell___healthy",
                    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
                    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight",
                    "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot",
                    "Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato__Target_Spot",
                    "Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato__Tomato_mosaic_virus",
                    "Tomato_healthy"};
            // CREATING A LIST THAT CONTSINS THE CONFIDENCCE VALUES FOR EACH CLASS FOR EACH MODEL
            float[] probabilities = outputFeature0lenet.getFloatArray();
            float[] probabilitiesvgg = outputFeature0vgg16.getFloatArray();
            float[] probabilitiesdensenet = outputFeature0densenet.getFloatArray();
            // Getting the highest conficdence level and obtaining its index
            int maxIndexlenet = argmax(probabilities);
            int maxIndexvgg = argmax(probabilitiesvgg);
            int maxIndexdensenet = argmax(probabilitiesdensenet);
            float maxValuelenet = probabilities[maxIndexlenet];
            float maxValuevgg = probabilities[maxIndexvgg];
            float maxValuedensenet = probabilities[maxIndexdensenet];
            // Calulating the percentage value for the class with highest confidence score for the models
            float sum = 0;
            for (float probability : probabilities) {
                sum += probability;
            }
            float maxPercentage = (maxValuelenet / sum) * 100;
            DecimalFormat df = new DecimalFormat("#.#");
            maxPercentage = Float.parseFloat(df.format(maxPercentage));

            float sum1 = 0;
            for (float probability1 : probabilitiesvgg) {
                sum1 += probability1;
            }
            float maxPercentage1 = (maxValuevgg / sum1) * 100;
            DecimalFormat df1 = new DecimalFormat("#.#");
            maxPercentage1 = Float.parseFloat(df1.format(maxPercentage1));

            float sum2 = 0;
            for (float probabilitydense : probabilitiesdensenet) {
                sum2 += probabilitydense;
            }
            float maxPercentage2 = (maxValuedensenet / sum2) * 100;
            DecimalFormat df2 = new DecimalFormat("#.#");
            maxPercentage2 = Float.parseFloat(df2.format(maxPercentage2));
            result.setText("LeNet5 " + classes[predictedClassIndexlenet] + " " + maxPercentage + "%.");
            result2.setText("VGG16 " + classes[predictedClassIndexvgg16] + " " + maxPercentage1 + "%.");
            result3.setText("DenseNet24 " + classes[predictedClassIndexdensenet] + " " + maxPercentage2 + "%.");

        } catch (Exception e) {
            // Handle the exception and log the error message
            Toast.makeText(this, "Error predicting image", Toast.LENGTH_SHORT).show();
            e.printStackTrace();
        }


    }
    private int argmax(float[] probabilities) {
        // Returning the maxindex for the confidence values
        int maxIndex = 0;
        float maxValue = probabilities[0];
        for (int i = 1; i < probabilities.length; i++) {
            if (probabilities[i] > maxValue) {
                maxIndex = i;
                maxValue = probabilities[i];
            }
        }
        return maxIndex;
    }



    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        // when user clicks capture button, the image they capture is resized and converted to a bitmap.
        if (resultCode == Activity.RESULT_OK) {
            if (requestCode == REQUEST_IMAGE_CAPTURE) {
                Bitmap imageBitmap = (Bitmap) data.getExtras().get("data");
                int dimension = Math.min(imageBitmap.getWidth(), imageBitmap.getHeight());
                imageBitmap = Bitmap.createScaledBitmap(imageBitmap, dimension, dimension, false);
                imagePlant.setImageBitmap(imageBitmap);
                imageBitmap = Bitmap.createScaledBitmap(imageBitmap, ImgSize, ImgSize, false);
                predictImage(imageBitmap);
            } else if (requestCode == REQUEST_IMAGE_FROM_GALLERY && data != null) {
                Uri selectedImageUri = data.getData();
                // If user clicks upload button and clicks an image the image is resized and converted into a bitmap and the predict image method is called passing the image
                try {
                    InputStream inputStream = getContentResolver().openInputStream(selectedImageUri);
                    Bitmap imageBitmap = BitmapFactory.decodeStream(inputStream);
                    int dimension = Math.min(imageBitmap.getWidth(), imageBitmap.getHeight());
                    imageBitmap = Bitmap.createScaledBitmap(imageBitmap, dimension, dimension, false);
                    imagePlant.setImageBitmap(imageBitmap);
                    imageBitmap = Bitmap.createScaledBitmap(imageBitmap, ImgSize, ImgSize, false);
                    predictImage(imageBitmap);
                } catch (Exception e) {
                    Toast.makeText(this, "Error loading image", Toast.LENGTH_SHORT).show();
                    e.printStackTrace();
                }
            }

        }
    }
    @Override
    protected void onDestroy() {
        super.onDestroy();
        // Close the interpreter to release resources
        if (tfliteInterpreterlenet != null) {
            tfliteInterpreterlenet.close();
        }
        if (tfliteInterpretervgg16 != null) {
            tfliteInterpretervgg16.close();
        }
        if (tfliteInterpreterdensenet != null) {
            tfliteInterpreterdensenet.close();
        }
    }

}