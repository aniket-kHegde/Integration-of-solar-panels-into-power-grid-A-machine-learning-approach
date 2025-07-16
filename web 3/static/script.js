document.addEventListener('DOMContentLoaded', function() {
    // Initialize map and variables
    let map, circle, buildingLayers;
    let currentTotalArea = 0;
    let currentAdjustedArea = 0;
    let window_size = 10;
  

    // Chart instances
    let irradianceChart, powerChart, demandChart, netPowerChart;

    try {
        // Initialize map with default coordinates that are definitely valid
        map = L.map('map').setView([12.92470, 77.50124], 13);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(map);

        // Initialize circle with validation
        const initialRadius = parseInt(document.getElementById('radius-slider').value);
        circle = L.circle([12.92470, 77.50124], {
            radius: initialRadius,
            draggable: true,
            color: '#3388ff',
            fillOpacity: 0.3
        }).addTo(map);

        // Initialize building layers
        buildingLayers = L.layerGroup().addTo(map);

        // Initialize charts
        initCharts();

        // Set up date picker
        const datePicker = document.getElementById('date-picker');
        const tomorrow = new Date();
        tomorrow.setDate(tomorrow.getDate() + 1);
        datePicker.min = new Date().toISOString().split('T')[0];
        datePicker.max = new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString().split('T')[0];
        datePicker.valueAsDate = tomorrow;

        // Event listeners with validation
        document.getElementById('refresh-btn').addEventListener('click', function() {
            if (validateCoordinates(circle.getLatLng())) {
                updateUI();
            }
        });

        document.getElementById('calculate-adjusted-area').addEventListener('click', calculateAdjustedArea);
        
        document.getElementById('analyze-btn').addEventListener('click', function() {
            if (validateCoordinates(circle.getLatLng())) {
                analyzeBuildings();
            }
        });
        
        document.getElementById('clear-btn').addEventListener('click', clearBuildings);
        
        document.getElementById('radius-slider').addEventListener('input', function() {
            const radius = parseInt(this.value);
	    document.getElementById('radius-value').textContent = radius;
            if (!isNaN(radius) && radius > 0 && radius <= 5000) {
                circle.setRadius(radius);
                if (validateCoordinates(circle.getLatLng())) {
                    updateUI();
                }
            }
        });

        circle.on('moveend', function() {
            if (validateCoordinates(circle.getLatLng())) {
                updateUI();
            }
        });

        map.on('click', function(e) {
            if (validateCoordinates(e.latlng)) {
                circle.setLatLng(e.latlng);
                updateUI();
            } else {
                alert("Invalid coordinates clicked. Please select a location within valid ranges.");
            }
        });

        // Initial UI update
        updateUI();

    } catch (e) {
        console.error("Initialization error:", e);
        alert("Failed to initialize application. Please refresh the page.");
    }

    function validateCoordinates(latlng) {
        try {
            const lat = parseFloat(latlng.lat);
            const lng = parseFloat(latlng.lng);
            return !isNaN(lat) && !isNaN(lng) && 
                   lat >= -90 && lat <= 90 && 
                   lng >= -180 && lng <= 180;
        } catch (e) {
            console.error("Coordinate validation error:", e);
            return false;
        }
    }

    function initCharts() {

        const chartOptions = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom' },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                    // Change units based on chart
                    let unit = 'W/m²'; // Default for irradiance
                    if (context.chart.canvas.id === 'powerChart' || 
                        context.chart.canvas.id === 'demandChart' || 
                        context.chart.canvas.id === 'netPowerChart') {
                        unit = 'kWh';
                    }
                    return `${context.dataset.label}: ${context.raw?.toFixed(2) || 0} ${unit}`;
                }
                    }
                }
            },
            scales: {
                y: { beginAtZero: true },
                x: { 
                    grid: { display: false },
                    ticks: {
                        callback: function(value) {
                            return `${value}:00`;
                        }
                    }
                }
            }
        };

        // Irradiance Chart
        const irradianceCtx = document.getElementById('irradianceChart').getContext('2d');
        irradianceChart = new Chart(irradianceCtx, {
            type: 'line',
            data: {
                labels: Array(24).fill().map((_,i) => i),
                datasets: [
                    {
                        label: 'GHI (W/m²)',
                        borderColor: '#FFA726',
                        backgroundColor: 'rgba(255, 167, 38, 0.1)',
                        fill: true,
                        tension: 0.3,
                        data: []
                    },
                    {
                        label: 'DNI (W/m²)',
                        borderColor: '#66BB6A',
                        tension: 0.3,
                        data: []
                    },
                    {
                        label: 'DHI (W/m²)',
                        borderColor: '#42A5F5',
                        tension: 0.3,
                        data: []
                    }
                ]
            },
            options: {
                ...chartOptions,
                plugins: {
                    ...chartOptions.plugins,
                    title: { display: true, text: 'Solar Irradiance', font: { size: 16 } }
                }
            }
        });

        // Power Chart
    const powerCtx = document.getElementById('powerChart').getContext('2d');
    powerChart = new Chart(powerCtx, {
        type: 'line',
        data: {
            labels: Array(24).fill().map((_,i) => i),
            datasets: [{
                label: 'Model Prediction (kWh)',
                borderColor: '#8e5ea2',
                backgroundColor: 'rgba(142, 94, 162, 0.1)',
                fill: true,
                tension: 0.3,
                data: []
            }]
        },
        options: {
            ...chartOptions,
            plugins: {
                ...chartOptions.plugins,
                title: { 
                    display: true, 
                    text: 'Solar Power Generation Prediction', 
                    font: { size: 16 } 
                }
            }
        }
    });

    // New Demand Chart
    const demandCtx = document.getElementById('demandChart').getContext('2d');
    demandChart = new Chart(demandCtx, {
        type: 'line',
        data: {
            labels: Array(24).fill().map((_,i) => i),
            datasets: [{
                label: 'Energy Demand (kWh)',
                borderColor: '#ff0000',
                backgroundColor: 'rgba(255, 0, 0, 0.1)',
                fill: true,
                tension: 0.3,
                data: []
            }]
        },
        options: {
            ...chartOptions,
            plugins: {
                ...chartOptions.plugins,
                title: { 
                    display: true, 
                    text: 'Energy Demand Curve', 
                    font: { size: 16 } 
                }
            }
        }
    });
const netPowerCtx = document.getElementById('netPowerChart').getContext('2d');
netPowerChart = new Chart(netPowerCtx, {
    type: 'line',
    data: {
        labels: Array(24).fill().map((_,i) => i),
        datasets: [{
            label: 'Net Power (kWh)',
            borderColor: '#4CAF50',
            backgroundColor: 'rgba(76, 175, 80, 0.1)',
            fill: true,
            tension: 0.3,
            data: []
        }]
    },
    options: {
        ...chartOptions,
        plugins: {
            ...chartOptions.plugins,
            title: { 
                display: true, 
                text: 'Net Power (Generation - Demand)', 
                font: { size: 16 } 
            }
        }
    }
});
}

    async function fetchSolarData(lat, lng) {
        try {
            if (!validateCoordinates({lat, lng})) {
                throw new Error("Invalid coordinates provided");
            }

            const response = await fetch('/get_solar', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    lat: parseFloat(lat.toFixed(6)),
                    lng: parseFloat(lng.toFixed(6)), 
                    date: document.getElementById('date-picker').value
                })
            });
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.error) throw new Error(data.error);
            if (!data.hourly || !Array.isArray(data.hourly)) throw new Error("Invalid data format");
            
            return data;
            
        } catch (error) {
            console.error('Error fetching solar data:', error);
            throw error;
        }
    }


function updateNetPowerChart() {
    try {
	if (!netPowerChart || !powerChart || !demandChart) return;
        // Get data from both charts
        const generationData = powerChart.data.datasets[0].data;
        const demandData = demandChart.data.datasets[0].data;
        
        // Calculate net power 
        const netPower = Array(24).fill().map((_, i) => {
            const gen = generationData[i] || 0;
            const dem = demandData[i] || 0;
            return dem - gen;
        });
        
        // Update the chart
        netPowerChart.data.datasets[0].data = netPower;
        netPowerChart.update();
    } catch (e) {
        console.error("Error updating net power chart:", e);
    }
}

    async function fetchBuildingData(lat, lng, radius) {
        try {
            if (!validateCoordinates({lat, lng})) {
                throw new Error("Invalid coordinates provided");
            }

            const numericRadius = parseInt(radius);
            if (isNaN(numericRadius) || numericRadius <= 0 || numericRadius > 5000) {
                throw new Error("Invalid radius value");
            }

            const response = await fetch('/get_buildings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    lat: parseFloat(lat.toFixed(6)),
                    lng: parseFloat(lng.toFixed(6)),
                    radius: numericRadius
                })
            });
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.error) throw new Error(data.error);
            
            return data;
            
        } catch (error) {
            console.error('Error fetching building data:', error);
            throw error;
        }
    }

async function predictPowerOutput(lat, lng, date, area) {
    try {
        // Get solar data
        const solarResponse = await fetch('/get_15min_solar', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ lat, lng, date })
        });
        
        if (!solarResponse.ok) throw new Error(`Solar data error: ${solarResponse.status}`);
        
        const solarData = await solarResponse.json();
        if (!solarData || !Array.isArray(solarData.hourly)) throw new Error("Invalid solar data format");

        // Build sequences - now we need 24 points per hour
        const hourlySequences = [];
        const allDataPoints = solarData.hourly;
        
        for (let hour = 0; hour < 24; hour++) {
            // Get all data points for this hour (should be 4 points at 15-min intervals)
            const hourData = allDataPoints.filter(item => item?.hour === hour);
            
            if (hourData.length === 0) {
                // If no data for this hour, create default values
                hourData.push(
                    { irradiation: 0, ambient_temp: 25, module_temp: 30 },
                    { irradiation: 0, ambient_temp: 25, module_temp: 30 },
                    { irradiation: 0, ambient_temp: 25, module_temp: 30 },
                    { irradiation: 0, ambient_temp: 25, module_temp: 30 }
                );
            } else if (hourData.length < 4) {
                // If we have some but not all 15-min intervals, fill with last available data
                while (hourData.length < 4) {
                    hourData.push({...hourData[hourData.length-1]});
                }
            }

            // Create sequence of 24 points by interpolating between the 15-min data points
            const sequence = [];
            
            // For each 15-minute interval, create 6 data points (24 total per hour)
            for (let i = 0; i < 4; i++) {
                const current = hourData[i];
                const next = hourData[(i + 1) % 4];
                
                // Add 6 points between each 15-minute interval
                for (let j = 0; j < 6; j++) {
                    const ratio = j / 6;
                    sequence.push([
                        (current.irradiation + (next.irradiation - current.irradiation) * ratio) / 1000, // Divide by 1000
                        current.ambient_temp + (next.ambient_temp - current.ambient_temp) * ratio,
                        current.module_temp + (next.module_temp - current.module_temp) * ratio
                    ]);
                }
            }
            
            hourlySequences.push(sequence);
        }
	const center = circle.getLatLng();
        // Get predictions
        const predictionResponse = await fetch('/predict_power', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
            lat: center.lat,
            lng: center.lng,
            date: document.getElementById('date-picker').value,
	    sequences: hourlySequences, area })
        });
        
        if (!predictionResponse.ok) {
            const error = await predictionResponse.json().catch(() => ({}));
            throw new Error(error.error || "Prediction failed");
        }
        
        const prediction = await predictionResponse.json();
        if (!prediction || !Array.isArray(prediction.hourly_predictions)) {
            throw new Error("Invalid prediction format");
        }
        
        return prediction;
        
    } catch (error) {
        console.error('Prediction error:', error);
        throw error;
    }
}


async function fetchAndPlotDemandCurve(month, area) {
    try {
        const response = await fetch('/get_demand_curve', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ month, area })
        });
        
        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.error || "Failed to fetch demand data");
        }
        
        const data = await response.json();
        if (!data.hourly_demand) {
            throw new Error("Invalid demand data format");
        }
        
        // Update demand chart
        const demandData = Array(24).fill(0);
        data.hourly_demand.forEach(item => {
            demandData[item.hour] = item.power_kw*currentTotalArea;
        });
        
        demandChart.data.datasets[0].data = demandData;
        demandChart.update();
	updateNetPowerChart();
        
    } catch (error) {
        console.error('Demand curve error:', error);
        // Clear demand chart on error
        demandChart.data.datasets[0].data = [];
        demandChart.update();
	updateNetPowerChart();
    }
}





    function updateIrradianceChart(data) {
        try {
            // Create hourly buckets
            const hourlyData = Array(24).fill().map(() => ({
                ghi: 0, dni: 0, dhi: 0, count: 0
            }));

            data.hourly.forEach(item => {
                const hour = item.hour;
                hourlyData[hour].ghi += item.ghi || 0;
                hourlyData[hour].dni += item.dni || 0;
                hourlyData[hour].dhi += item.dhi || 0;
                hourlyData[hour].count++;
            });

            // Update chart
            irradianceChart.data.datasets[0].data = hourlyData.map(h => h.count ? h.ghi/h.count : 0);
            irradianceChart.data.datasets[1].data = hourlyData.map(h => h.count ? h.dni/h.count : 0);
            irradianceChart.data.datasets[2].data = hourlyData.map(h => h.count ? h.dhi/h.count : 0);
            
            irradianceChart.update();
        } catch (e) {
            console.error("Error updating irradiance chart:", e);
        }
    }

    function updatePowerChart() {
        try {
	    if (!powerChart) return;
            const area = currentAdjustedArea || 0;
            // Calculate basic estimate from irradiance data
            if (irradianceChart?.data?.datasets[0]?.data) {
                powerChart.data.datasets[0].data = irradianceChart.data.datasets[0].data.map(ghi => 
                    Math.max(0, (ghi || 0) * area / 1000) // Convert to kW, no efficiency
                );
                powerChart.update();
		updateNetPowerChart();
            }
        } catch (e) {
            console.error("Error updating power chart:", e);
        }
    }

    function calculateAdjustedArea() {
        try {
            const percentage = parseFloat(document.getElementById('percentage-input').value) || 100;
            currentAdjustedArea = (currentTotalArea * percentage / 100) || 0;
            document.getElementById('adjusted-area').textContent = 
                `Adjusted area: ${currentAdjustedArea.toFixed(2)} m² (${percentage}%)`;
            updatePowerChart();
        } catch (e) {
            console.error("Error calculating adjusted area:", e);
        }
    }

    async function analyzeBuildings() {
        try {
            const center = circle.getLatLng();
            const radius = circle.getRadius();
            
            const data = await fetchBuildingData(center.lat, center.lng, radius);
            
            if (data.error) {
                alert(data.error);
                return;
            }
            
            updateBuildingUI(data);
            drawBuildingsOnMap(data.buildings);
            updatePowerChart();
        } catch (error) {
            console.error('Error analyzing buildings:', error);
            alert('Error analyzing buildings: ' + error.message);
        }
    }

    function updateBuildingUI(data) {
        try {
            document.getElementById('building-count').textContent = `Buildings found: ${data.count || 0}`;
            document.getElementById('total-area').textContent = `Total roof area: ${(data.total_roof_area || 0).toFixed(2)} m²`;
            
            const avgArea = (data.count > 0) ? (data.total_roof_area / data.count) : 0;
            document.getElementById('avg-area').textContent = `Average roof area: ${avgArea.toFixed(2)} m²`;
            
            currentTotalArea = data.total_roof_area || 0;
            currentAdjustedArea = data.total_roof_area || 0;
            document.getElementById('adjusted-area').textContent = 
                `Adjusted area: ${currentAdjustedArea.toFixed(2)} m² (100%)`;
            
            // Update building details
            const detailsContainer = document.getElementById('building-details');
            if (data.buildings && data.buildings.length > 0) {
                detailsContainer.innerHTML = data.buildings
                    .map(b => `<div>Building ID: ${b.id || 'N/A'} - Area: ${(b.area || 0).toFixed(2)} m²</div>`)
                    .join('');
            } else {
                detailsContainer.textContent = 'No building data available';
            }
        } catch (e) {
            console.error("Error updating building UI:", e);
        }
    }

    function drawBuildingsOnMap(buildings) {
        try {
            buildingLayers.clearLayers();
            
            if (buildings && Array.isArray(buildings)) {
                buildings.forEach(building => {
                    if (building.coordinates && Array.isArray(building.coordinates)) {
                        const polygon = L.polygon(building.coordinates, {
                            color: '#3388ff',
                            fillOpacity: 0.3,
                            weight: 2
                        }).bindPopup(`Area: ${(building.area || 0).toFixed(2)} m²`);
                        
                        buildingLayers.addLayer(polygon);
                    }
                });
            }
        } catch (e) {
            console.error("Error drawing buildings:", e);
        }
    }

    function clearBuildings() {
        try {
            buildingLayers.clearLayers();
            document.getElementById('building-count').textContent = 'Buildings found: 0';
            document.getElementById('total-area').textContent = 'Total roof area: 0 m²';
            document.getElementById('avg-area').textContent = 'Average roof area: 0 m²';
            document.getElementById('adjusted-area').textContent = 'Adjusted area: 0 m²';
            document.getElementById('building-details').textContent = 'No building data available';
            currentTotalArea = 0;
            currentAdjustedArea = 0;
            updatePowerChart();
        } catch (e) {
            console.error("Error clearing buildings:", e);
        }
    }

    async function updateUI() {
        try {
            // Initialize charts if not already done
            if (!irradianceChart || !powerChart || !demandChart || !netPowerChart) {
            initCharts();
        }

            const center = circle.getLatLng();
            
            // Validate coordinates before proceeding
            if (!validateCoordinates(center)) {
                throw new Error("Invalid map coordinates");
            }

            document.getElementById('center-coords').textContent = 
                `${center.lat.toFixed(5)}, ${center.lng.toFixed(5)}`;
            
            // Get and display solar data
            const solarData = await fetchSolarData(center.lat, center.lng);
            document.getElementById('forecast-date').textContent = 
                `Showing data for: ${new Date(solarData.date).toDateString()}`;
            
            updateIrradianceChart(solarData);
            updatePowerChart();
            
            // Get model prediction if area is selected
        if (currentAdjustedArea > 0) {
            try {
                document.getElementById('model-prediction').textContent = 
                    'Model prediction: Calculating...';
                
                const prediction = await predictPowerOutput(
                    center.lat, 
                    center.lng, 
                    document.getElementById('date-picker').value,
                    currentAdjustedArea
                );
                
                // Update model prediction display
                const totalKW = prediction.total_prediction;
                document.getElementById('model-prediction').textContent = 
                    `Model prediction: ${totalKW.toFixed(2)} kW`;
                
                // Update power chart with predictions
                powerChart.data.datasets[0].data = prediction.hourly_predictions;
                powerChart.update();
		updateNetPowerChart();
                
                // Get and plot demand curve
                const date = new Date(document.getElementById('date-picker').value);
                const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
                const month = monthNames[date.getMonth()];
                await fetchAndPlotDemandCurve(month, currentAdjustedArea);
                
            } catch (error) {
                console.error('Model prediction error:', error);
                document.getElementById('model-prediction').textContent = 
                    `Model prediction: ${error.message}`;
                powerChart.data.datasets[0].data = [];
                powerChart.update();
            }
        } else {
                document.getElementById('model-prediction').textContent = 
                    'Model prediction: Select an area first';
                powerChart.data.datasets[1].data = [];
                powerChart.update();
            }
            
        } catch (error) {
            console.error('UI update error:', error);
            document.getElementById('model-prediction').textContent = 
                'Model prediction: Data loading failed';
            //alert('Error loading data: ' + error.message);
        }
    }
});