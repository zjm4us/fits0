#include "TRandom2.h"
#include "TGraphErrors.h"
#include "TMath.h"
#include "TApplication.h"
#include "TCanvas.h"
#include "TH2F.h"
#include "TH1F.h"
#include "TF1.h"
#include "TMinuit.h"
#include "TStyle.h"

#include <iostream>
using namespace std;

using TMath::Log;

// parms
const double xmin = 1;
const double xmax = 20;
const int npoints = 12;
const double sigma = 0.2;

// model function
double f(double x) {
  const double a = 0.5;
  const double b = 1.3;
  const double c = 0.5;
  return a + b * Log(x) + c * Log(x) * Log(x);
}

// Generate x values evenly spaced
void getX(double *x) {
  double step = (xmax - xmin) / npoints;
  for (int i = 0; i < npoints; i++) {
    x[i] = xmin + i * step;
  }
}

// Generate y values with Gaussian noise, and errors ey = sigma
void getY(const double *x, double *y, double *ey) {
  static TRandom2 tr(0);
  for (int i = 0; i < npoints; i++) {
    y[i] = f(x[i]) + tr.Gaus(0, sigma);
    ey[i] = sigma;
  }
}

// --- Fit function to minimize chi2 ---
double fitfunc(double *xx, double *par) {
  // par[0]=a, par[1]=b, par[2]=c
  double x = xx[0];
  return par[0] + par[1] * Log(x) + par[2] * Log(x) * Log(x);
}

// --- Chi2 function for Minuit ---
void chi2Function(int &npar, double *gin, double &f, double *par, int iflag) {
  static double x[npoints], y[npoints], ey[npoints];
  static bool firstCall = true;

  if (firstCall) {
    getX(x);
    getY(x, y, ey);
    firstCall = false;
  }

  f = 0.;
  for (int i = 0; i < npoints; i++) {
    double yi = par[0] + par[1] * Log(x[i]) + par[2] * Log(x[i]) * Log(x[i]);
    double diff = (y[i] - yi) / ey[i];
    f += diff * diff;
  }
}

int main(int argc, char **argv) {
  // TApplication theApp("App", &argc, argv); // No GUI needed, comment out

  // ******************************************************************************
  // ** this block is useful for supporting both high and std resolution screens **
  //UInt_t dh = gClient->GetDisplayHeight() / 2;   // fix plot to 1/2 screen height  
  //UInt_t dw = gClient->GetDisplayWidth();
  //UInt_t dw = 1.1 * dh;
  // ******************************************************************************

  gStyle->SetOptStat(0); // turn off histogram stats box

  // Prepare data for one pseudo experiment and draw it (save instead of draw)
  double lx[npoints];
  double ly[npoints];
  double ley[npoints];

  getX(lx);
  getY(lx, ly, ley);

  auto tgl = new TGraphErrors(npoints, lx, ly, 0, ley);
  tgl->SetTitle("Pseudoexperiment;x;y");

  TCanvas *tc = new TCanvas("c1", "Sample dataset", 800, 600);
  tgl->Draw("alp");
  tc->SaveAs("pseudoexperiment_cpp.png");  // Save plot as image instead of drawing interactively

  // --- Histograms to store fit parameters and chi2 ---
  TH2F *h1 = new TH2F("h1", "Parameter b vs a;a;b", 100, 0, 1, 100, 0, 2);
  TH2F *h2 = new TH2F("h2", "Parameter c vs a;a;c", 100, 0, 1, 100, 0, 1);
  TH2F *h3 = new TH2F("h3", "Parameter c vs b;b;c", 100, 0, 2, 100, 0, 1);
  TH1F *h4 = new TH1F("h4", "reduced chi^2;;frequency", 100, 0, 5);

  // perform many least squares fits on different pseudo experiments here
  // fill histograms w/ required data

  const int nPseudo = 1000;  // Number of pseudo experiments

  TRandom2 randGen(0);
  double x[npoints], y[npoints], ey[npoints];

  for (int iPE = 0; iPE < nPseudo; iPE++) {
    getX(x);
    for (int i = 0; i < npoints; i++) {
      y[i] = f(x[i]) + randGen.Gaus(0, sigma);
      ey[i] = sigma;
    }

    // Setup Minuit for fitting a,b,c
    TMinuit minuit(3);
    minuit.SetPrintLevel(-1); // suppress output

    // Chi2 function for Minuit with data inside lambda closure
    // Since we can't capture easily, we pass pointers in a static way

    // Setup chi2 function with current data
    // We'll define a local chi2Function with current data inside here:
    struct FitData {
      double* x;
      double* y;
      double* ey;
      int n;
    } fitData = {x, y, ey, npoints};

    // Define chi2Function for Minuit that uses fitData:
    // Use the global chi2Function but set data pointers before calling
    // For simplicity, define a lambda function and wrap it in global

    static double* fitX;
    static double* fitY;
    static double* fitEY;
    static int fitN;

    fitX = x;
    fitY = y;
    fitEY = ey;
    fitN = npoints;

    auto minuitChi2 = [](int &npar, double *gin, double &f, double *par, int iflag) {
      f = 0.;
      for (int i = 0; i < fitN; i++) {
        double yi = par[0] + par[1] * Log(fitX[i]) + par[2] * Log(fitX[i]) * Log(fitX[i]);
        double diff = (fitY[i] - yi) / fitEY[i];
        f += diff * diff;
      }
    };

    // Register the chi2 function
    minuit.SetFCN(minuitChi2);

    // Set initial parameters a, b, c (guesses)
    double parStart[3] = {0.5, 1.0, 0.5};
    double parStep[3] = {0.01, 0.01, 0.01};
    const char* parName[3] = {"a", "b", "c"};

    for (int i = 0; i < 3; i++) {
      minuit.DefineParameter(i, parName[i], parStart[i], parStep[i], 0, 0);
    }

    // Do the minimization
    minuit.Migrad();

    // Get fitted parameters and errors
    double a_fit, a_err;
    double b_fit, b_err;
    double c_fit, c_err;

    minuit.GetParameter(0, a_fit, a_err);
    minuit.GetParameter(1, b_fit, b_err);
    minuit.GetParameter(2, c_fit, c_err);

    // Calculate reduced chi2
    double chi2_val;
    minuit.Eval(3, 0, chi2_val, parStart, 0); // get chi2 with initial params

    // But better calculate chi2 with fitted params:
    chi2_val = 0.;
    for (int i = 0; i < npoints; i++) {
      double yi = a_fit + b_fit * Log(x[i]) + c_fit * Log(x[i]) * Log(x[i]);
      double diff = (y[i] - yi) / ey[i];
      chi2_val += diff * diff;
    }
    double red_chi2 = chi2_val / (npoints - 3);

    // Fill histograms
    h1->Fill(a_fit, b_fit);
    h2->Fill(a_fit, c_fit);
    h3->Fill(b_fit, c_fit);
    h4->Fill(red_chi2);
  }

  TCanvas *tc2 = new TCanvas("c2", "my study results", 800, 600);
  tc2->Divide(2, 2);

  tc2->cd(1);
  h1->Draw("colz");
  tc2->cd(2);
  h2->Draw("colz");
  tc2->cd(3);
  h3->Draw("colz");
  tc2->cd(4);
  h4->Draw();

  tc2->SaveAs("study_results_cpp.png");  // Save final results as image

  cout << "Completed 1000 pseudoexperiments and saved results to PNG files." << endl;

  // no interactive app->Run() since no GUI

  return 0;
}

